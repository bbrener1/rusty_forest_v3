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

use crate::io::{ParameterBook,read};
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


pub trait Sample: Clone + Debug
{
    type Prototype: Prototype<Value=Self::Value>;
    type Key: SampleKey;
    type Value: SampleValue;

    fn from(index:usize,prototype:Self::Prototype) -> Self;
    fn index(&self) -> usize;
    fn prototype<'a>(&'a self) -> &'a Self::Prototype;

    fn output_slice(&self) -> ArrayView1<Self::Value>{
        self.prototype().output_array().slice(s![self.index(),..])
    }

    fn input_slice(&self) -> ArrayView1<Self::Value> {
        self.prototype().input_array().slice(s![self.index(),..])
    }

    fn name(&self) -> &str {
        &self.prototype().parameters().sample_names[self.index()]
    }

    fn output_feature<F:OutputFeature>(&self,feature:&F) -> Self::Value {
        self.output_slice()[feature.index()]
    }

    fn input_feature<F:InputFeature>(&self,feature:&F) -> Self::Value {
        self.input_slice()[feature.index()]
    }


}

pub trait Feature : Clone + Debug
{
    type Prototype: Prototype<Value=Self::Value>;
    type Sample: Sample;
    type Key: FeatureKey;
    type Value: SampleValue;

    fn index(&self) -> usize;
    fn prototype(&self) -> &Self::Prototype;
    fn slice(&self) -> ArrayView1<Self::Value>;

    fn sample(&self,sample:Self::Sample) -> Self::Value {
        self.slice()[sample.index()]
    }

}

pub trait InputFeature : Feature {
    type Prototype: Prototype<InputFeature=Self>;

    fn sorted_indices(&self) -> ArrayView1<usize> {
        self.prototype().sorted_index_array().slice(s![self.index(),..])
    }
    fn slice(&self) -> ArrayView1<Self::Value> {
        self.prototype().input_array().slice(s![..,self.index()])
    }

}

pub trait OutputFeature : Feature {
    fn slice(&self) -> ArrayView1<Self::Value> {
        self.prototype().output_array().slice(s![..,self.index()])
    }
}

pub trait Prototype : Clone + Debug
where
{
    type InputFeature: InputFeature;
    type OutputFeature: OutputFeature;
    type Sample: Sample;
    type Value: SampleValue;

    fn input_array(&self) -> &Array2<Self::Value>;
    fn output_array(&self) -> &Array2<Self::Value>;
    // fn input_features(&self) -> &[Self::InputFeature];
    // fn output_features(&self) -> &[Self::OutputFeature];
    fn parameters(&self) -> &ParameterBook<Self::Value>;
    fn sorted_index_array(&self) -> &Array2<usize>;
    fn double_select_output(&self,samples:&[Self::Sample],features:&[Self::OutputFeature]) -> Array2<Self::Value> {
        let mut selection = Array2::zeros((samples.len(),features.len()));
        for (i,si) in samples.iter().map(|s| s.index()).enumerate() {
            for (j,fj) in features.iter().map(|f| f.index()).enumerate() {
                selection[[i,j]] = self.output_array()[[si,fj]]
            }
        }
        selection
    }
}

#[derive(Clone,Debug)]
pub struct UFPrototype<V:SampleValue> {
    inputs: Array2<V>,
    outputs: Array2<V>,
    sorted_input_indices: Array2<usize>,
    parameters: ParameterBook<V>,
}

impl<'a,V:SampleValue> UFPrototype<V> {
    fn from_parameters(parameters:ParameterBook<V>) -> UFPrototype<V> {
        let inputs = parameters.input_array.take().unwrap();
        let outputs = parameters.output_array.take().unwrap();
        let mut sorted_input_indices: Array2<usize> = Array2::zeros((inputs.dim().1,inputs.dim().0));
        for (i,feature) in inputs.axis_iter(Axis(1)).enumerate() {
            let valsorted: Vec<(usize,&V)> = valsort(feature.iter());
            for (j,(si,_)) in valsorted.into_iter().enumerate() {
                sorted_input_indices[[i,j]]=si;
            }
            // sorted_input_indices.row_mut(i).assign(&valsorted.iter().map(|x| x.0).collect());
        }

        UFPrototype {
            inputs,
            outputs,
            sorted_input_indices,
            parameters
        }
    }
}

impl<'a,V:SampleValue> Prototype for UFPrototype<V> {
    type InputFeature = InputFeatureUF<'a,V>;
    type OutputFeature = OutputFeatureUF<'a,V>;
    type Sample = SampleUF<'a,V>;
    type Value = V;

    fn input_array(&self) -> &Array2<Self::Value> {
        &self.inputs
    }
    fn output_array(&self) -> &Array2<Self::Value> {
        &self.outputs
    }
    // fn input_features(&'a self) -> &[Self::InputFeature] {
    //     &self.input_features()
    // }
    // fn output_features(&self) -> &[Self::OutputFeature]{
    //     &self.output_featues
    // }
    fn parameters(&self) -> &ParameterBook<Self::Value> {
        &self.parameters
    }
    fn sorted_index_array(&self) -> &Array2<usize> {
        &self.sorted_input_indices
    }

}

#[derive(Clone,Debug)]
pub struct SampleUF<'a,V:SampleValue> {
    index: usize,
    prototype: &'a UFPrototype<V>
}

impl<'a,V:SampleValue> Sample for SampleUF<'a,V> {

    type Prototype = UFPrototype<V>;
    type Key = usize;
    type Value = V;

    fn from(index:usize,prototype:UFPrototype<V>) -> Self {
        SampleUF {
            index:index,
            prototype:&prototype,
        }
    }
    fn index(&self) -> usize {
        self.index
    }

    fn prototype(&self) -> &Self::Prototype {
        &self.prototype
    }

}

#[derive(Clone,Debug)]
pub struct InputFeatureUF<'a,V:SampleValue> {
    index: usize,
    prototype: &'a UFPrototype<V>
}

impl<'a,V:SampleValue> InputFeatureUF<'a,V> {

    fn from(index:usize,prototype: &'a UFPrototype<V>) -> Self {
        InputFeatureUF {
            index: index,
            prototype: prototype,
        }
    }
}

impl<'a,V:SampleValue> Feature for InputFeatureUF<'a,V> {
    type Prototype = UFPrototype<V>;
    type Sample = SampleUF<'a,V>;
    type Key = usize;
    type Value = V;

    fn index(&self) -> usize {
        self.index
    }
    fn prototype(&self) -> &Self::Prototype {
        &self.prototype
    }

    fn slice(&self) -> ArrayView1<Self::Value> {
        InputFeature::slice(self)
    }
}

impl<'a,V:SampleValue> InputFeature for InputFeatureUF<'a,V> {
    type Prototype = UFPrototype<V>;
}

#[derive(Clone,Debug)]
pub struct OutputFeatureUF<'a,V: SampleValue> {
    index: usize,
    prototype: &'a UFPrototype<V>
}

impl<'a,V:SampleValue> OutputFeatureUF<'a,V> {
    fn from(index:usize,prototype: &'a UFPrototype<V>) -> OutputFeatureUF<V> {
        OutputFeatureUF {
            index: index,
            prototype: prototype,
        }
    }
}

impl<'a,V:SampleValue> Feature for OutputFeatureUF<'a,V> {
    type Prototype = UFPrototype<V>;
    type Sample = SampleUF<'a,V>;
    type Key = usize;
    type Value = V;

    fn index(&self) -> usize {
        self.index
    }
    fn prototype(&self) -> &Self::Prototype {
        &self.prototype
    }

    fn slice(&self) -> ArrayView1<Self::Value> {
        OutputFeature::slice(self)
    }
}

impl<'a,V:SampleValue> OutputFeature for OutputFeatureUF<'a,V> {}

fn valsort<I: Iterator<Item=T>,T: PartialOrd + Clone>(s:I) -> Vec<(usize,T)>{
    let mut paired: Vec<(usize,T)> = s.into_iter().cloned().enumerate().collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    paired
}
