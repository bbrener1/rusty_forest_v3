
use std::sync::Arc;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::sync::mpsc;
use std::f64;
use std::mem::replace;
use std::collections::{HashMap,HashSet};
use serde_json;

extern crate rand;
use rand::Rng;

use rayon::prelude::*;

use crate::{Feature,InputFeature,OutputFeature,Forest};
use crate::Sample;
use crate::SampleKey;
use crate::FeatureKey;
use crate::SampleValue;
use crate::Prototype;
use crate::SampleFilter;
use crate::subsample;
// use crate::io::DispersionMode;
use crate::io::ParameterBook;
use crate::rank_vector::{SegmentedVector,FeatureVector};
use crate::nipals::calculate_projection;
use crate::valsort;
use crate::rank_vector::MedianArray;
use std::iter::Map;
use crate::argmax;
use crate::ArgMinMax;
use nipals::Projector;

use rayon::prelude::*;

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

    fn derive(&self,SampleFilter<Self::InputFeature>) -> Self;

    fn split(&mut self,depth:usize) -> &mut Self {

        if depth > self.parameters().depth_cutoff || self.samples().len() > self.parameters().leaf_size_cutoff {
            return self
        }

        // println!("SPLITTING");

        let input_feature_subsample = self.forest().subsample_input_features();
        let output_feature_subsample = self.forest().subsample_output_features();
        let (in_bag,out_bag) = self.sample_bags();
        let sample_subsample = subsample(&in_bag, self.forest().parameters().sample_subsample);
        // let sample_subsample: Vec<&Self::Sample> = subsample(self.samples(), self.forest().parameters().sample_subsample);
        // let sample_subsample: Vec<&Self::Sample> = subsample(&samples, self.forest().parameters().sample_subsample);
        // let input_feature_subsample: Vec<Self::InputFeature> = self.forest().input_features().to_vec();
        // let output_feature_subsample: Vec<Self::OutputFeature> = self.forest().output_featues().to_vec();
        // let sample_subsample: Vec<Self::Sample> = self.samples().to_vec();

        // println!("SUBSAMPLED");

        let input_intermediate = self.prototype().double_select_input(&sample_subsample,&input_feature_subsample);
        let output_intermediate = self.prototype().double_select_output(&sample_subsample,&output_feature_subsample);
        // let out_of_bag_intermediate = self.prototype().double_select_output(&out_of_bag,&output_feature_subsample);
        // let (reduction,reduced_intermediate) = calculate_projection(output_intermediate);
        let (reduced_intermediate,reduction,means) = Projector::from(output_intermediate).calculate_projection();
        let valsorted: Vec<(usize,f64)> = valsort(reduced_intermediate.to_vec().into_iter());

        // println!("REDUCED");
        // println!("{:?}",reduction);
        // println!("{:?}", reduced_intermediate);
        // println!("{:?}", valsorted);

        let mut mv = MedianArray::link(&valsorted);
        // println!("LINKED");
        // println!("{:?}", mv);

        let sample_indices = sample_subsample.iter().map(|s| s.index()).collect();
        let mut cached_sorter = CachedSorter::from(sample_indices);

        // println!("CACHED");
        // println!("{:?}",cached_sorter);
        let mut dispersions = vec![0.;cached_sorter.draw_order.len()];

        let sfr = self.prototype().parameters().split_fraction_regularization;
        let (best_feature,(best_sample,dispersion)) = input_feature_subsample.into_iter()
        .map(|f| {
            // println!("FEATURE:{:?}",f.index());
            let si = f.sorted_indices();
            // println!("{:?}", si);
            cached_sorter.sort(si.iter());
            // println!("SORTED:{:?}",cached_sorter);
            // println!("SORTED+VAL:{:?}", cached_sorter.draw_order.iter().map(|i| InputFeature::slice(&f)[*i]).collect::<Vec<Self::Value>>());
            let ss_len = sample_subsample.len();
            let mut mv_f = mv.clone();
            let mut mv_r = mv.clone();
            for (i,draw) in cached_sorter.draw_order.iter().enumerate() {
                // println!("POSITION:{:?}",i);
                mv_f.pop(*draw);
                let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
                {dispersions[i] = mv_f.dispersion() * regularization;}
                // println!("{:?}",dispersions[i]);
            }
            for (i,draw) in cached_sorter.draw_order.iter().rev().enumerate() {
                // println!("POSITION:{:?}",ss_len - i -1);
                mv_r.pop(*draw);
                let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
                {dispersions[ss_len - i -1] += mv_r.dispersion() * regularization;}
                // println!("{:?}",dispersions[ss_len - i -1]);

            }
            // println!("FINISHED DISPERSIONS:{:?}",dispersions);
            if let Some((best_split,&dispersion)) = dispersions.iter().argmin_v()
            {
                let best_sample_local_index = cached_sorter.draw_order[best_split];
                let best_sample = sample_subsample[best_sample_local_index].clone();
                Some((f,(best_sample,dispersion)))
            }
            else {None}

// TODO CHECK TO MAKE SURE YOU GET THE CORRECT ORDER DISPERSIONS

                // F64 Minimum because Rust gonna Rust
        })
        .flat_map(|x| x)
        .min_by(|a,b| (a.1).1.partial_cmp(&(b.1).1).unwrap()).unwrap();

        // println!("BEST FEATURE/SAMPLE: {:?},{:?}",best_feature,best_sample);

        let (left_fitler,right_filter) = SampleFilter::from_feature_sample(&best_feature, &best_sample);

        let (left_child,right_child) = (self.derive(left_fitler),self.derive(right_filter));
        self.mut_children().push(left_child);
        self.mut_children().push(right_child);

        for child in self.mut_children() {
            child.split(depth+1);
        }

        self
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
        handle.write(serde_json::to_string(self).ok()?.as_bytes());
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
    fn derive(&self,filter:SampleFilter<InputFeatureUF<V>>) -> FastNode<'a,V> {
        let new_samples = filter.filter_samples(&self.samples);
        FastNode {
            samples: new_samples,
            forest: self.forest,
            parameters: self.parameters,
            filter: filter,
            children: vec![],
        }
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
            if let Some((minimum_rank,quantity)) = self.stencil.get_mut(key) {
                self.draw_order[*minimum_rank] = ss_index;
                *minimum_rank += 1;
            }
        }
    }

    fn draw_order(&self) -> &[usize] {
        &self.draw_order
    }
}
