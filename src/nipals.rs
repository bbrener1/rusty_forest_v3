use std::f64;
use std::convert::Into;
use num_traits::Num;
use ndarray::prelude::*;

use crate::SampleValue;


pub fn calculate_projection<T:SampleValue>(input:Array2<T>) -> (Array1<f64>,Array1<f64>) {
    let smallnum = 10e-4 ;
    let mut converted:Array2<f64> = input.map(|v| T::into(*v));
    converted = center(converted);
    let mut loadings = Array::ones(converted.dim().0);
    let mut scores = Array::ones(converted.dim().1);
    let mut score_norm = f64::MAX;
    loop {
        scores = converted.t().dot(&loadings);
        let new_norm = (scores.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt();
        let delta = (score_norm - new_norm).abs();
        scores.mapv_inplace(|x| x/new_norm);
        if delta < smallnum {
            return (scores,loadings)
        }
        else {score_norm = new_norm};
        loadings = converted.dot(&scores);
    }
}

fn center(mut input:Array2<f64>) -> Array2<f64> {
    let means = input.mean_axis(Axis(0));
    for mut row in input.axis_iter_mut(Axis(0)) {
        row -= &means;
    }
    input
}
