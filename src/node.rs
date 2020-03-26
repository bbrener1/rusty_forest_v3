
use std::cmp::PartialOrd;
use std::f64;
use std::collections::{HashMap};
use serde_json;

extern crate rand;

use rayon::prelude::*;

use crate::{InputFeature,OutputFeature,Forest,Reduction};
use crate::Sample;
use crate::SampleValue;
use crate::Prototype;
use crate::SampleFilter;
use crate::subsample;
// use crate::io::DispersionMode;
use crate::io::ParameterBook;
use crate::rank_vector::{SegmentedVector,FeatureVector};
use crate::valsort;
use crate::rank_vector::MedianArray;
use crate::ArgMinMax;
use nipals::Projector;
use ndarray::Array1;

use crate::{PrototypeUF,InputFeatureUF,OutputFeatureUF,SampleUF,ForestUF,SerializedFilter};

pub trait Node<'a> : Clone
{
    type Forest: Forest<Prototype=Self::Prototype,Value=Self::Value,Sample=Self::Sample,InputFeature=Self::InputFeature,OutputFeature=Self::OutputFeature>;
    type Prototype: Prototype<Value=Self::Value,Sample=Self::Sample,InputFeature=Self::InputFeature,OutputFeature=Self::OutputFeature>;
    type Value: SampleValue;
    type Sample: Sample<Prototype=Self::Prototype,Value=Self::Value>;
    type InputFeature: InputFeature<Prototype=Self::Prototype,Sample=Self::Sample,Value=Self::Value>;
    type OutputFeature: OutputFeature<Prototype=Self::Prototype,Value=Self::Value,Sample=Self::Sample>;

    fn from_forest(forest:&'a Self::Forest) -> Self;
    fn filter(&self) -> &SampleFilter<Self::InputFeature>;
    fn forest(&self) -> &Self::Forest;
    fn samples(&self) -> & [Self::Sample];
    fn samples_mut(&mut self) -> & mut [Self::Sample];
    fn prototype(&self) -> &Self::Prototype;
    fn children(&self) -> & Vec<Self>;
    fn mut_children(&mut self) -> &mut Vec<Self>;
    fn parameters(&self) -> &'a ParameterBook<Self::Value>;

    fn to_sidxn(&self) -> SampleIndexNode {
        let samples = self.samples().iter().map(|s| s.index()).collect();
        let children = self.children().iter().map(|c| c.to_sidxn()).collect();
        SampleIndexNode{
            samples,
            filter: self.filter().serialize().clone(),
            children
        }
    }

}

pub trait ComputeNode<'a>: Node<'a>
{

    fn derive(&self,SampleFilter<Self::InputFeature>) -> Option<Self>;

    fn split(&mut self,depth:usize) -> Option<&mut Self> {
        use num_traits::{NumCast};

        if depth > self.parameters().depth_cutoff || self.samples().len() < self.parameters().leaf_size_cutoff {
            return None
        }

        // println!("SPLITTING");

        let input_feature_subsample: Vec<Self::InputFeature> = self.forest().subsample_input_features();
        let output_feature_subsample: Vec<Self::OutputFeature> = self.forest().subsample_output_features();
        let (in_bag,out_bag) = self.sample_bags();
        let sample_subsample = subsample(&in_bag, self.forest().parameters().sample_subsample);
        // let sample_subsample: Vec<&Self::Sample> = subsample(self.samples(), self.forest().parameters().sample_subsample);
        // let sample_subsample: Vec<&Self::Sample> = subsample(&samples, self.forest().parameters().sample_subsample);
        // let input_feature_subsample: Vec<Self::InputFeature> = self.forest().input_features().to_vec();
        // let output_feature_subsample: Vec<Self::OutputFeature> = self.forest().output_featues().to_vec();
        // let sample_subsample: Vec<Self::Sample> = self.samples().to_vec();

        // println!("SUBSAMPLED");

        // let input_intermediate = self.prototype().double_select_input(&sample_subsample,&input_feature_subsample);
        let output_intermediate = self.prototype().double_select_output(&sample_subsample,&output_feature_subsample);
        let (reduced_output,output_scores,output_means,_) = Projector::from(output_intermediate).calculate_projection();
        let valsorted: Vec<(usize,f64)> = valsort(reduced_output.to_vec().into_iter());

        let output_reduction =
            Reduction::from(
                output_feature_subsample,
                output_scores.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect(),
                output_means.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect());

        // println!("REDUCED");
        // println!("{:?}",reduction);
        // println!("{:?}", reduced_intermediate);
        // println!("{:?}", valsorted);

        let mv = MedianArray::link(&valsorted);
        // println!("LINKED");
        // println!("{:?}", mv);

        let sample_indices = sample_subsample.iter().map(|s| s.index()).collect();
        let cached_sorter = CachedSorter::from(sample_indices);

        // println!("CACHED");
        // println!("{:?}",cached_sorter);

        let sfr = self.prototype().parameters().split_fraction_regularization;
        let (best_feature,(best_sample,_)) = input_feature_subsample
        // .into_iter()
        .into_par_iter()
        .map(|f| {
            // println!("FEATURE:{:?}",f.index());
            let si = f.sorted_indices();
            // println!("{:?}", si);
            let mut local_sorter = cached_sorter.clone();
            let mut dispersions = vec![0.;cached_sorter.draw_order.len()];
            // let mut local_sorter = &mut cached_sorter;
            local_sorter.sort(si.iter());
            // println!("SORTED:{:?}",cached_sorter);
            // println!("SORTED+VAL:{:?}", cached_sorter.draw_order.iter().map(|i| InputFeature::slice(&f)[*i]).collect::<Vec<Self::Value>>());
            let ss_len = sample_subsample.len();
            let mut mv_f = mv.clone();
            let mut mv_r = mv.clone();
            for (i,draw) in local_sorter.draw_order.iter().enumerate() {
                // println!("POSITION:{:?}",i);
                mv_f.pop(*draw);
                let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
                {dispersions[i] = mv_f.dispersion() * regularization;}
                // println!("{:?}",dispersions[i]);
            }
            for (i,draw) in local_sorter.draw_order.iter().rev().enumerate() {
                // println!("POSITION:{:?}",ss_len - i -1);
                mv_r.pop(*draw);
                let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
                {dispersions[ss_len - i -1] += mv_r.dispersion() * regularization;}
                // println!("{:?}",dispersions[ss_len - i -1]);

            }
            // println!("FINISHED DISPERSIONS:{:?}",dispersions);
            if let Some((best_split,&dispersion)) = dispersions.iter().argmin_v()
            {
                let best_sample_local_index = local_sorter.draw_order[best_split];
                let best_sample = sample_subsample[best_sample_local_index].clone();
                Some((f,(best_sample,dispersion)))
            }
            else {None}

                // F64 Minimum because Rust gonna Rust
        })
        .flat_map(|x| x)
        .min_by(|a,b| (a.1).1.partial_cmp(&(b.1).1).unwrap()).unwrap();

        // println!("BEST FEATURE/SAMPLE: {:?},{:?}",best_feature,best_sample);

        let (left_fitler,right_filter) = SampleFilter::from_feature_sample(&best_feature, &best_sample);

        let left_oob = left_fitler.filter_samples(&out_bag);
        let right_oob = right_filter.filter_samples(&out_bag);

        if let (Some(left_child),Some(right_child)) = (self.derive(left_fitler),self.derive(right_filter)) {
            self.mut_children().push(left_child);
            self.mut_children().push(right_child);

            for child in self.mut_children() {
                child.split(depth+1);
            }

            Some(self)
        }
        else { Some(self) }

    }

    fn double_reduce(&mut self,depth:usize) -> Option<&mut Self> {
        use num_traits::{NumCast};

        if depth > self.parameters().depth_cutoff || self.samples().len() < self.parameters().leaf_size_cutoff {
            return None
        }

        let input_feature_subsample: Vec<Self::InputFeature> = self.forest().subsample_input_features();
        let output_feature_subsample: Vec<Self::OutputFeature> = self.forest().subsample_output_features();
        let (in_bag,out_bag) = self.sample_bags();
        let sample_subsample = subsample(&in_bag, self.forest().parameters().sample_subsample);


        let input_intermediate = self.prototype().double_select_input(&sample_subsample,&input_feature_subsample);
        let output_intermediate = self.prototype().double_select_output(&sample_subsample,&output_feature_subsample);
        let (reduced_input,input_scores,input_means,input_scales) = Projector::from(input_intermediate).calculate_projection();
        let (reduced_output,output_scores,output_means,output_scales) = Projector::from(output_intermediate).calculate_projection();
        let reduced_input = reduced_input / input_scales;
        let reduced_output = reduced_output / output_scales;


        let input_reduction =
            Reduction::from(
                input_feature_subsample,
                input_scores.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect(),
                input_means.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect());

        let output_reduction =
            Reduction::from(
                output_feature_subsample,
                output_scores.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect(),
                output_means.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect());

        let valsorted_input: Vec<(usize,f64)> = valsort(reduced_input.to_vec().into_iter());
        let valsorted_output: Vec<(usize,f64)> = valsort(reduced_output.to_vec().into_iter());

        let draw_order: Vec<usize> = valsorted_input.into_iter().map(|(i,_)| i).collect();
        let mut dispersions = vec![0.;draw_order.len()];
        let mv = MedianArray::link(&valsorted_output);

        let sfr = self.prototype().parameters().split_fraction_regularization;

        let ss_len = sample_subsample.len();
        let mut mv_f = mv.clone();
        let mut mv_r = mv.clone();

        for (i,draw) in draw_order.iter().enumerate() {
            mv_f.pop(*draw);
            let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
            {dispersions[i] = mv_f.dispersion() * regularization;}
        }
        for (i,draw) in draw_order.iter().rev().enumerate() {
            mv_r.pop(*draw);
            let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
            {dispersions[ss_len - i -1] += mv_r.dispersion() * regularization;}

        }
        //     // println!("FINISHED DISPERSIONS:{:?}",dispersions);
        let (best_split,_) = dispersions.iter().argmin_v()?;
        let local_index = draw_order[best_split];
        let split = NumCast::from(reduced_input[local_index]).expect("Cast failure");

        let (left_fitler,right_filter) = SampleFilter::from_reduction(input_reduction, split);
        //
        if let (Some(left_child),Some(right_child)) = (self.derive(left_fitler),self.derive(right_filter)) {
            self.mut_children().push(left_child);
            self.mut_children().push(right_child);

            for child in self.mut_children() {
                child.double_reduce(depth+1);
            }

            Some(self)
        }
        else { Some(self) }

    }

    fn check_predictions(filter:&SampleFilter<Self::InputFeature>,output_reduction:&Reduction<Self::OutputFeature>,in_bag:&[Self::Sample],out_bag:&[Self::Sample])
    {
        let filtered_inbag = filter.filter_samples(in_bag);
        let filtered_outbag = filter.filter_samples(out_bag);

        let reduced_inbag: Array1<Self::Value> = filtered_inbag.iter().map(|s| output_reduction.transform_sample(s)).collect();
        let sorted_reduced_inbag:Vec<(usize,Self::Value)> = valsort(reduced_inbag.to_vec().into_iter());
        let mv = MedianArray::link(&sorted_reduced_inbag);
        let prediction = mv.central_tendency();

        let reduced_outbag: Array1<Self::Value> = filtered_outbag.iter().map(|s| output_reduction.transform_sample(s)).collect();

    }

    fn sample_bags(&self) -> (Vec<Self::Sample>,Vec<Self::Sample>) {
        use rand::seq::SliceRandom;
        use rand::prelude::thread_rng;

        let mut in_bag: Vec<Self::Sample> = self.samples().to_vec();
        let bag_size = (in_bag.len() as f64 * 0.66) as usize;
        &mut in_bag.partial_shuffle(&mut thread_rng(),bag_size);
        let out_bag = in_bag.split_off(bag_size);
        (in_bag,out_bag)
    }

}

#[derive(Clone,Debug,Serialize)]
pub struct SampleIndexNode {
    samples: Vec<usize>,
    filter: SerializedFilter,
    children: Vec<SampleIndexNode>,
}

impl SampleIndexNode
{
    pub fn dump(&mut self,filename:&str) -> Option<()> {
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut handle = OpenOptions::new().write(true).truncate(true).create(true).open(filename).ok()?;
        handle.write(serde_json::to_string(self).ok()?.as_bytes()).ok()?;
        Some(())
    }
}

#[derive(Clone,Debug)]
pub struct FastNode<'a,V:SampleValue> {
    forest: &'a ForestUF<V>,
    parameters: &'a ParameterBook<V>,
    samples: Vec<SampleUF<V>>,
    filter: SampleFilter<InputFeatureUF<V>>,
    children: Vec<FastNode<'a,V>>,
}

impl<'a,V:SampleValue> Node<'a> for FastNode<'a,V> {
    type Forest = ForestUF<V>;
    type Prototype = PrototypeUF<V>;
    type Value = V;
    type Sample = SampleUF<V>;
    type InputFeature = InputFeatureUF<V>;
    type OutputFeature = OutputFeatureUF<V>;

    fn from_forest(forest:&'a ForestUF<V>) -> FastNode<V> {
        FastNode {
            samples: forest.samples().to_vec(),
            forest: forest,
            parameters: forest.parameters(),
            filter: SampleFilter::<InputFeatureUF<V>>::blank(),
            children: vec![],
        }
    }

    fn filter(&self) -> &SampleFilter<Self::InputFeature> {
        &self.filter
    }

    fn samples(&self) -> & [SampleUF<V>] {
        &self.samples
    }

    fn samples_mut(&mut self) -> &mut [SampleUF<V>] {
        &mut self.samples
    }

    fn children(&self) -> &Vec<Self> {
        &self.children
    }

    fn mut_children(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    fn forest(&self) -> &ForestUF<V> {
        &self.forest
    }

    fn parameters(&self) -> &'a ParameterBook<V> {
        &self.parameters
    }

    fn prototype(&self) -> &PrototypeUF<V> {
        self.forest.prototype()
    }
}

impl<'a,V:SampleValue> ComputeNode<'a> for FastNode<'a,V> {
    fn derive(&self,filter:SampleFilter<InputFeatureUF<V>>) -> Option<FastNode<'a,V>> {
        let new_samples = filter.filter_samples_scaled(&self.samples);
        if new_samples.len() > 0 {
            Some(FastNode {
                samples: new_samples,
                forest: self.forest,
                parameters: self.parameters,
                filter: filter,
                children: vec![],
            })
        }
        else { None }
    }
}

#[derive(Clone,Debug)]
struct CachedSorter {
    subsample: Vec<usize>,
    stencil: HashMap<usize,(usize,usize)>,
    draw_order: Vec<usize>,
}

impl CachedSorter {
    fn from(subsample:Vec<usize>) -> Self {

        // First we count how many times each key occurrs in the subsample;

        let mut stencil_map: HashMap<usize,(usize,usize)> = HashMap::with_capacity(subsample.len());
        for i in subsample.iter() {
            stencil_map.entry(*i).or_insert((0,0)).1 += 1;
        }
        CachedSorter {
            draw_order: vec![subsample.len();subsample.len()],
            subsample:subsample,
            stencil: stencil_map,
         }
    }

    fn sort<'a,I:ExactSizeIterator<Item=&'a usize>>(&mut self,sorted_indices:I) {

        // For a given sort order, we must now determine the minimum rank of each present index.
        // We can do this by iterating through the sorted indices and keeping track of the quantity
        // seen so far.

        let mut current_subsample_rank = 0;
        for key in sorted_indices {
            if let Some((minimum_rank,quantity)) = self.stencil.get_mut(&(key)) {
                *minimum_rank = current_subsample_rank;
                current_subsample_rank += *quantity;
            }
        }

        // For each sample key present in the subsample we now know the minimum rank and quantity.
        // We proceed through the subsample and insert the subsample index at a the minimum rank of
        // that subsample


        for (ss_index,key) in self.subsample.iter().enumerate() {
            if let Some((minimum_rank,_)) = self.stencil.get_mut(key) {
                self.draw_order[*minimum_rank] = ss_index;
                *minimum_rank += 1;
            }
        }
    }

    fn draw_order(&self) -> &[usize] {
        &self.draw_order
    }
}
