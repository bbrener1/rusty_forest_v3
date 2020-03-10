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
use std::fmt::{Debug};
use num_traits::{Zero,One,Num,FromPrimitive,ToPrimitive,NumCast,Pow,Bounded,Signed};
use std::str::FromStr;
use std::ops::{SubAssign,AddAssign,Sub};
use std::iter::Sum;
use std::sync::Arc;
use std::convert::Into;

use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::HashMap;

use crate::io::{ParameterBook,read,Parameters};
use std::env;


fn main() {
    let mut arg_iter = env::args();

    let mut parameters = read(&mut arg_iter);



}

pub trait SampleKey: Hash + Eq + Copy + Clone + Debug + FromStr {}

impl SampleKey for usize {}
// impl SampleKey for &str {}

pub trait FeatureKey: Hash + Eq + Clone + Debug + FromStr {}

impl FeatureKey for String {}
impl FeatureKey for usize {}

pub trait SampleValue: Num + Zero + FromStr + Clone + Copy + Into<f64> + LinalgScalar + Debug + PartialOrd + SubAssign + AddAssign + Signed + FromPrimitive + Sum + Bounded + ToPrimitive + NumCast + Pow<u8,Output=Self> {}

impl SampleValue for f64 {}
impl SampleValue for f32 {}
impl SampleValue for i32 {}

//
// pub trait Sample: Clone + Debug
// {
//     type Prototype: Prototype<Value=Self::Value>;
//     type Key: SampleKey;
//     type Value: SampleValue;
//
//     fn index(&self) -> usize;
//     fn prototype<'a>(&'a self) -> &'a Self::Prototype;
//     fn key(&self) -> Self::Key;
//
//     fn output_slice(&self) -> ArrayView1<Self::Value>{
//         self.prototype().output_array().slice(s![self.index(),..])
//     }
//
//     fn input_slice(&self) -> ArrayView1<Self::Value> {
//         self.prototype().input_array().slice(s![self.index(),..])
//     }
//
//     fn name(&self) -> &str {
//         &self.prototype().parameters().sample_names()[self.index()]
//     }
//     fn output_feature<F:Feature>(&self,feature:&F) -> Self::Value {
//         self.output_slice()[feature.index()]
//     }
// }
//
// pub trait Feature : Clone + Debug
// {
//     type Sample: Sample;
//     type Key: FeatureKey;
//     type Value: SampleValue;
//
//     fn index(&self) -> usize;
//     fn name(&self) -> String;
//     fn sample(&self,sample:&Self::Sample) -> Self::Value;
//
//     fn samples(&self,samples:&[Self::Sample]) -> &[Self::Value];
//     fn sorted_indices(&self) -> &[usize];
//     fn slice(&self) -> &[Self::Value];
// }

pub trait Prototype : Clone + Debug
where
{
    type Feature: FeatureKey;
    type Sample: SampleKey;
    type Value: SampleValue;
    type Parameters: Parameters;

    fn input_array(&self) -> &ArrayView2<Self::Value>;
    fn output_array<'a>(&'a self) -> &ArrayView2<'a,Self::Value>;
    fn input_features(&self) -> &[Self::Feature];
    fn output_features(&self) -> &[Self::Feature];
    fn samples(&self) -> &[Self::Sample];

    fn parameters(&self) -> &Self::Parameters;

    fn sorted_index_array(&self,feature:Self::Feature) -> ArrayView1<Self::Value>;

    fn double_select_output(&self,samples:&[Self::Sample],features:&[Self::Feature]) -> Array2<Self::Value>;
}

pub struct ArcUFPrototype<'a> {
    features: Vec<FeatureS<'a>>,
    samples: Vec<SampleU>,
    parameters: Arc<ParameterBook<f64>>
}



fn valsort<T: PartialOrd + Clone>(s:&[T]) -> Vec<(usize,T)>{
    let mut paired: Vec<(usize,T)> = s.into_iter().cloned().enumerate().collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    paired
}
