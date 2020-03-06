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

use std::hash::Hash;
use std::cmp::{Eq};
use std::fmt::Debug;
use num_traits::{Zero,One,Num,FromPrimitive,ToPrimitive,NumCast,Pow};
use std::ops::{SubAssign,AddAssign};

fn main() {
    println!("Hello, world!");
}

pub trait SampleKey: Hash + Eq + Clone + Copy + Debug {}
pub trait FeatureKey: Hash + Eq + Clone + Copy + Debug {}
pub trait SampleValue: Num + Zero + Clone + Copy + Debug + SubAssign + AddAssign + FromPrimitive + ToPrimitive + NumCast + Pow<i8,Output=Self>{}

impl SampleKey for usize {}

impl SampleValue for f64 {}

pub trait Sample
{
    type Key: SampleKey;
    type Value: SampleValue;

    fn feature<F:Feature>(&self,feature:&F) -> Self::Value;
    fn features<F:Feature>(&self,features:&[F]) -> &[Self::Value];
    fn slice(&self) -> &[Self::Value];
}

pub trait Feature
{
    type Key: FeatureKey;
    type Value: SampleValue;
    fn sample<S:Sample>(&self,sample:&S) -> Self::Value;
    fn samples<S:Sample>(&self,samples:&[S]) -> &[Self::Value];
    fn slice(&self) -> &[Self::Value];
}

pub trait Prototype
where
{
    type Feature: Feature;
    type Sample: Sample;
    fn sort_by_feature(&self,feature:Self::Feature) -> &[Self::Sample];
}
