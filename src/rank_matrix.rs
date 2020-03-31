use ndarray::prelude::*;
use crate::rank_vector::{FeatureVector,SegmentedVector,LinkedVector};
use crate::{SampleKey,SampleValue,DrawOrder};
use crate::rank_vector::MedianArray;
use crate::valsort;
use crate::ArgMinMax;

pub fn split<V:SampleValue>(input:&Array2<V>,output:&Array2<f64>,sfr:f64) -> Option<(usize,usize)> {

    let mut output_vectors: Vec<MedianArray<f64>> = vec![];
    for column in output.axis_iter(Axis(1)) {
        let valsorted = valsort(column.iter().cloned());
        output_vectors.push(MedianArray::link(&valsorted))
    }

    let mut draw_orders: Vec<Vec<usize>> = vec![];
    for column in input.axis_iter(Axis(1)) {
        draw_orders.push(valsort(column.iter()).into_iter().map(|(i,_)| i).collect());
    }

    let mut minima = vec![];
    let ss_len = output.dim().0;

    for (i,draw_order) in draw_orders.iter().enumerate() {
        let mut dispersions = vec![0.;draw_order.len()];
        for mut rv in output_vectors.iter_mut() {
            let mut rv_f = rv.clone();
            for (j,index) in draw_order.iter().enumerate() {
                rv_f.pop(*index);
                let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
                dispersions[j] += rv_f.dispersion()
            }
            let mut rv_r = rv.clone();
            for (j,index) in draw_order.iter().rev().enumerate() {
                rv_r.pop(*index);
                let regularization = ((ss_len - i) as f64 / ss_len as f64).powf(sfr);
                dispersions[draw_order.len() - j - 1] += rv_r.dispersion() * regularization;
            }
        }
        minima.push(dispersions.into_iter().argmin_v());
    }
    let (split_feature,_) = minima.iter().flat_map(|o| o).map(|(i,v)| v).argmin_v()?;
    let split_draw_order = &draw_orders[split_feature];
    let split_index = minima[split_feature]?.0;
    let split_sample = split_draw_order[split_index];

    Some((split_feature,split_sample))
}


#[cfg(test)]
mod matrix_tests {

    use crate::iris_array;
    use super::*;

    #[test]
    fn split_test() {
        let input = iris_array();
        let output = iris_array();
        println!("{:?}",split(&input,&output,0.));
        panic!();
    }

}
