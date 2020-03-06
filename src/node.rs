
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


use rayon::prelude::*;

trait Node
{
    type Value: SampleValue;
    type Sample: Sample;
    type Feature: Feature;

    fn samples(&self) -> &[Self::Sample];
    fn features(&self) -> &[Self::Feature];
}

trait ComputeNode: Node
{
    type ComputeVector: FeatureVector;
    fn split(&mut self);
}

trait StoredNode: Node
{
    fn dump(&mut self,filename:&str);
}

struct FastNode {
    prototype:Arc<Prototype>,
    output:
}
