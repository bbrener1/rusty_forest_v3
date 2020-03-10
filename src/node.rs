
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
// use crate::io::DispersionMode;
use crate::rank_vector::FeatureVector;
use crate::nipals::calculate_projection;
use crate::valsort;
use crate::rank_vector::MedianArray;

use rayon::prelude::*;

trait Node : Clone
{
    type Value: SampleValue;
    type Sample: SampleKey;
    type Feature: FeatureKey;
    type Prototype: Prototype<Value=Self::Value,Sample=Self::Sample,Feature=Self::Feature,Parameters=Self::Parameters>;
    type Parameters: Parameters;


    fn prototype(&self) -> Arc<Self::Prototype>;
    fn samples(&self) -> &[Self::Sample];
    fn input_features(&self) -> &[Self::Feature];
    fn output_features(&self) -> &[Self::Feature];
    fn stencil(&self) -> &[usize];

    fn parameters(&self) -> &Self::Parameters {
        self.prototype().parameters()
    }
}

trait ComputeNode: Node
{
    //
    fn split(&mut self) {

        let input_feature_subsample = fast_subsample(self.input_features(), self.parameters().input_feature_subsample());
        let output_feature_subsample = fast_subsample(self.output_features(), self.parameters().output_feature_subsample());
        let sample_subsample = fast_subsample(self.samples(), self.parameters().sample_subsample());

        let output_intermediate = self.prototype().double_select_output(&sample_subsample,&output_feature_subsample);
        let (reduction,reduced_intermediate) = calculate_projection(output_intermediate);
        let valsorted = valsort(reduced_intermediate.slice(s![..]).into_slice().unwrap());
        let mut mv = MedianArray::link(&valsorted);

        let draw_order_iterators = self.draw_order_iterators(&input_feature_subsample);
        for doi in draw_order_iterators.into_iter() {

        }


    }

    fn draw_order_iterators<'a>(&'a self,input_features:&'a[Self::Feature]) -> Vec<CachedSorter<'a>> {
        let mut draw_orders = Vec::with_capacity(input_features.len());
        for feature in input_features {
            let sorted = feature.sorted_indices();
            let draw_order = CachedSorter::from(sorted, self.stencil());
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

fn fast_subsample<T:Clone>(collection:&[T],draws:usize) -> Vec<T> {
    use rand::distributions::{Distribution, Uniform};
    let mut new_collection = Vec::with_capacity(draws);
    let between = Uniform::from(0..collection.len());
    let mut rng = rand::thread_rng();
    for _ in 0..draws {
        new_collection.push(collection[between.sample(&mut rng)].clone());
    }
    new_collection

}


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

struct CachedSorter<'a> {
    sorted_indices: &'a [usize],
    stencil: &'a [usize],
    current:usize,
    stencil_cache: usize
}

impl<'a> CachedSorter<'a> {
    fn from(sorted_indices:&'a [usize],stencil:&'a [usize]) -> Self {
        CachedSorter {
            sorted_indices: sorted_indices,
            stencil: stencil,
            current: 0,
            stencil_cache: stencil[sorted_indices[0]],
        }
    }
}

impl<'a> Iterator for CachedSorter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item>{
        while self.stencil_cache < 1 {
            self.current +=1;
            if self.current >= self.sorted_indices.len() {return None};
            self.stencil_cache = self.stencil[self.sorted_indices[self.current]];
        }
        let index = self.sorted_indices[self.current];
        self.stencil_cache -= 1;
        return Some(index)
    }

}
