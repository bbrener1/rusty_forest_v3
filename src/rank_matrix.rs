use ndarray::prelude::*;
use crate::rank_vector::{FeatureVector,SegmentedVector};
use crate::{SampleKey,SampleValue,DrawOrder};
use crate::rank_vector::MedianArray;
use crate::valsort;
use crate::ArgMinMax;

pub fn split<V:SampleValue>(input:&Array2<V>,output:&Array2<f64>,sfr:f64) -> Option<(usize,usize)> {

    let mut output_vectors: Vec<MedianArray<f64>> = vec![];
    for row in output.axis_iter(Axis(0)) {
        let valsorted = valsort(row.iter().cloned());
        output_vectors.push(MedianArray::link(&valsorted));
        // println!("Output vector:{:?}",output_vectors.last().unwrap());
    }

    let mut draw_orders: Vec<Vec<usize>> = vec![];
    for row in input.axis_iter(Axis(0)) {
        draw_orders.push(valsort(row.iter()).into_iter().map(|(i,_)| i).collect());
        // println!("Draw order:{:?}",draw_orders.last().unwrap().len());
    }

    let mut minima = vec![];
    let ss_len = output.dim().1;

    for (i,draw_order) in draw_orders.iter().enumerate() {
        let mut dispersions = vec![0.;draw_order.len()];
        for (k,rv) in output_vectors.iter().enumerate() {
            let mut rv_f = rv.clone();
            for (j,index) in draw_order.iter().enumerate() {
                rv_f.pop(*index);
                let regularization = ((ss_len - j) as f64 / ss_len as f64).powf(sfr);
                dispersions[j] += rv_f.dispersion()
            }
            let mut rv_r = rv.clone();
            for (j,index) in draw_order.iter().rev().enumerate() {
                rv_r.pop(*index);
                let regularization = ((ss_len - j) as f64 / ss_len as f64).powf(sfr);
                dispersions[ss_len - j - 1] += rv_r.dispersion() * regularization;
            }
        }
        let minimum = dispersions.into_iter().argmin_v().map(|(local_index,dispersion)| (i,draw_order[local_index],dispersion));
        minima.push(minimum);
    }
    let (feature,sample,dispersion) = minima.iter().flat_map(|m| m).min_by(|&a,&b| (a.2).partial_cmp(&b.2).unwrap())?;

    Some((*feature,*sample))
}


#[cfg(test)]
mod matrix_tests {

    use crate::iris_array;
    use super::*;

    #[test]
    fn split_test() {
        let input = iris_array().t().to_owned();
        let output = iris_array().t().to_owned();
        println!("{:?}",split(&input,&output,0.));
        panic!();
    }

}
