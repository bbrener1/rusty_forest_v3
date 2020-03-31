use ndarray::prelude::*;
use crate::rank_vector::{FeatureVector,SegmentedVector,LinkedVector};
use crate::{SampleKey,SampleValue,DrawOrder};
use crate::rank_vector::MedianArray;
use crate::valsort;
use crate::ArgMinMax;

fn split<V:SampleValue>(input:&Array2<V>,output:&Array2<V>) -> Option<(usize,usize)> {

    let mut output_vectors: Vec<MedianArray<V>> = vec![];
    for column in output.axis_iter(Axis(1)) {
        let valsorted = valsort(column.iter().cloned());
        output_vectors.push(MedianArray::link(&valsorted))
    }

    let mut draw_orders: Vec<Vec<usize>> = vec![];
    for column in input.axis_iter(Axis(1)) {
        draw_orders.push(valsort(column.iter()).into_iter().map(|(i,_)| i).collect());
    }

    let mut minima = vec![];

    for (i,draw_order) in draw_orders.iter().enumerate() {
        let mut dispersions = vec![V::zero();draw_order.len()];
        for mut rv in output_vectors.iter_mut() {
            let mut rv_f = rv.clone();
            for (j,index) in draw_order.iter().enumerate() {
                rv_f.pop(*index);
                dispersions[j] += rv_f.dispersion()
            }
            let mut rv_r = rv.clone();
            for (j,index) in draw_order.iter().rev().enumerate() {
                rv_r.pop(*index);
                dispersions[draw_order.len() - j - 1] += rv_r.dispersion()
            }
        }
        minima.push(dispersions.into_iter().argmin_v());
    }
    let (split_feature,_) = minima.iter().flat_map(|o| o).map(|(i,v)| v).argmin_v()?;
    let split_index = minima[split_feature]?.0;
    Some((split_feature,split_index))
}
