
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

use crate::Feature;
use crate::Sample;
use crate::SampleKey;
use crate::FeatureKey;
use crate::SampleValue;
use crate::Prototype;
use crate::io::Parameters;
use crate::io::DispersionMode;
use crate::rank_vector::FeatureVector;
use crate::nipals::calculate_projection;
use crate::valsort;
use crate::rank_vector::MedianArray;

use rayon::prelude::*;

trait Node
{
    type Value: SampleValue;
    type Sample: Sample;
    type Feature: Feature;
    type Prototype: Prototype;

    fn prototype(&self) -> Arc<Self::Prototype>;
    fn samples(&self) -> &[Self::Sample];
    fn features(&self) -> &[Self::Feature];
}

trait ComputeNode: Node
{
    type ComputeVector: FeatureVector;
    //
    // fn split(&mut self) {
    //     let input_feature_subsample = vec![];
    //     let output_feature_subsample = vec![];
    //     let sample_subsample = vec![];
    //     let output_intermediate = self.prototype().double_select(&sample_subsample,&output_feature_subsample);
    //     let (reduction,reduced_intermediate) = calculate_projection(output_intermediate.view());
    //     let valsorted = valsort(reduced_intermediate.slice(s![..]).into_slice().unwrap());
    //     let draw_orders = input_feature_subsample.into_iter().map(|f| self.prototype().sort_by_feature)
    //     let mut mv = MedianArray::link(&valsorted);
    //
    // }
    fn sample_subsample(available:&[Self::Sample],draws:usize) -> Vec<Self::Sample> {
        let mut new_samples = Vec::with_capacity(draws);
        use rand::distributions::{Distribution, Uniform};

        let mut stencil = vec![0;draws];
        let between = Uniform::from(0..available.len());
        let mut rng = rand::thread_rng();
        for _ in 0..draws {
            new_samples.push(available[between.sample(&mut rng)].clone());
        }
        new_samples
    }


    fn fast_draw_order_subsample(&self,available_samples:&[Self::Sample],input_features:&[Self::Feature],sample_subsample:usize) -> Vec<Vec<usize>> {
        let mut draw_orders = Vec::with_capacity(input_features.len());
        let available_sample_indices: Vec<usize> = available_samples.iter().map(|s| s.index()).collect();
        let stencil = stencil(&available_sample_indices,self.prototype().samples().len(),sample_subsample);
        for feature in input_features {
            let sorted = feature.sorted_indices();
            let draw_order = cached_sort_subsample(sorted, &stencil, sample_subsample);
            draw_orders.push(draw_order);
        }
        draw_orders
    }

}

trait StoredNode: Node
{
    fn dump(&mut self,filename:&str);
}
//
// struct FastNode {
//     prototype:Arc<Prototype>,
//     output:
// }


fn stencil(indices:&[usize],stencil_size:usize,draws:usize) -> Vec<usize> {
    let mut stencil = vec![0;stencil_size];
    for i in indices {
        stencil[*i] += 1;
    }
    stencil
}

fn cached_sort_subsample(sorted_indices:&[usize],stencil:&[usize],draws:usize) -> Vec<usize> {
    let mut output = vec![0;draws];
    let mut current = 0;
    for i in sorted_indices {
        for j in 0..stencil[*i] {
            output[current] = *i;
            current += 1;
        }
    }
    output
}
