// #![feature(test)]

// extern crate test;
// use test::Bencher;
//
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate ndarray;

extern crate serde;
extern crate serde_json;
extern crate num_cpus;
extern crate rand;
extern crate time;
extern crate smallvec;
extern crate rayon;

extern crate num_traits;

mod rank_vector;
mod io;
mod node;
mod nipals;

use std::hash::Hash;
use std::cmp::{Eq,PartialOrd};
use std::fmt::Debug;
use num_traits::{Zero,One,Num,FromPrimitive,ToPrimitive,NumCast,Pow,Bounded,Signed};
use std::ops::{SubAssign,AddAssign,Sub};
use std::iter::Sum;
use std::sync::Arc;

use ndarray::prelude::*;
use std::collections::HashMap;

fn main() {
    println!("Hello, world!");
}

pub trait SampleKey: Hash + Eq + Clone + Copy + Debug {}
pub trait FeatureKey: Hash + Eq + Clone + Copy + Debug {}
pub trait SampleValue: Num + Zero + Clone + Copy + Debug + PartialOrd + SubAssign + Sub<f64> + AddAssign + Signed + FromPrimitive + Sum + Bounded + ToPrimitive + NumCast + Pow<i8,Output=Self> + Pow<f32, Output=Self>{}

impl SampleKey for usize {}

impl SampleValue for f64 {}
//
pub trait Sample: Clone
{
    type Prototype: Prototype;
    type Key: SampleKey;
    type Value: SampleValue;

    fn prototype(&self) -> Self::Prototype;
    fn index(&self) -> usize;
    fn feature<F:Feature>(&self,feature:&F) -> Self::Value;
    fn features<F:Feature>(&self,features:&[F]) -> &[Self::Value];
    fn slice(&self) -> &[Self::Value];
}

pub trait Feature
{
    type Sample: Sample;
    type Key: FeatureKey;
    type Value: SampleValue;

    fn sample(&self,sample:&Self::Sample) -> Self::Value;
    fn samples(&self,samples:&[Self::Sample]) -> &[Self::Value];
    fn sorted_indices(&self) -> &[usize];
    fn slice(&self) -> &[Self::Value];
}

pub trait Prototype
where
{
    type Feature: Feature;
    type Sample: Sample;
    type Value: SampleValue;

    fn array(&self) -> ArrayView2<Self::Value>;
    fn features(&self) -> &[Self::Feature];
    fn samples(&self) -> &[Self::Sample];

    fn sorted_index_array(&self,feature:Self::Feature) -> ArrayView1<Self::Value>;

    fn double_select(&self,samples:&[Self::Sample],features:&[Self::Feature]) -> Array2<Self::Value>;
}


//
// pub struct Sample<K: SampleKey> {
//     prototype: Arc<Prototype<>>,
//     key: K,
// }
//
// pub struct Prototype<V>
// where
//     V: SampleValue,
// {
//     array:Array2<V>,
//     sorted_indices:Array2<usize>,
//     features: Vec<Feature>,
//     feature_map: HashMap<FeatureKey>
// }


fn valsort<T: PartialOrd + Clone>(s:&[T]) -> Vec<(usize,T)>{
    let mut paired: Vec<(usize,T)> = s.into_iter().cloned().enumerate().collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    paired
}
