use std::f64;
use ndarray::prelude::*;

use crate::SampleValue;


// TODO: Bugcheck when a 1x1 is passed in

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

#[derive(Clone,Debug)]
pub struct Projector {
    means: Array1<f64>,
    scale_factors: Array1<f64>,
    converted:Array2<f64>,
    scores:Array1<f64>,
    loadings:Array1<f64>,
    score_norm: f64,
    smallnum: f64,
}

impl Projector {
    pub fn from<T:SampleValue>(input:Array2<T>) -> Projector {
        let smallnum = 10e-4 ;
        let mut converted:Array2<f64> = input.map(|v| T::into(*v));
        let means = converted.mean_axis(Axis(0));
        let scale_factors = converted.sum_axis(Axis(1));
        converted = center(converted);
        let loadings = Array::ones(converted.dim().0);
        let scores = Array::ones(converted.dim().1);
        let score_norm = f64::MAX;
        Projector {
            means,
            scale_factors,
            converted,
            scores,
            loadings,
            score_norm,
            smallnum,
        }
    }

    pub fn calculate_projection(&mut self) -> (Array1<f64>,Array1<f64>,Array1<f64>,Array1<f64>) {
        self.scores.fill(1.);
        self.loadings.fill(1.);
        loop {
            self.scores = self.converted.t().dot(&self.loadings);
            let new_norm = (self.scores.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt();
            let delta = (self.score_norm - new_norm).abs();
            self.scores.mapv_inplace(|x| x/new_norm);
            if delta < self.smallnum {
                self.scores *= -1.;
                self.loadings *= -1.;
                self.converted -= &outer(&self.loadings,&self.scores);
                return (self.loadings.clone(),self.scores.clone(),self.means.clone(),self.scale_factors.clone())
            }
            else {self.score_norm = new_norm};
            self.loadings = self.converted.dot(&self.scores);
        }
    }

    pub fn calculate_n_projections(mut self,n:usize) -> (Array2<f64>,Array2<f64>,Array2<f64>,Array2<f64>) {
        let mut loadings = Array2::zeros((n,self.converted.dim().0));
        let mut scores = Array2::zeros((n,self.converted.dim().1));
        let mut means = Array2::zeros((n,self.converted.dim().1));
        let mut scale_factors = Array2::zeros((n,self.converted.dim().0));
        for i in 0..n {
            // println!("Projection {:?}",i);
            // println!("{:?}",self);
            let (n_loadings,n_scores,n_means,n_scale_factors)= self.calculate_projection();
            loadings.row_mut(i).assign(&n_loadings);
            scores.row_mut(i).assign(&n_scores);
            means.row_mut(i).assign(&n_means);
            scale_factors.row_mut(i).assign(&n_scale_factors);
        }
        (loadings,scores,means,scale_factors)
    }

}



fn outer(v1:&Array1<f64>,v2:&Array1<f64>) -> Array2<f64> {
    let m = v1.len();
    let n = v2.len();
    let mut output = Array2::zeros((m,n));
    for mut row in output.axis_iter_mut(Axis(0)) {
        row.assign(v2);
    }
    for mut column in output.axis_iter_mut(Axis(1)) {
        column *= v1;
    }
    output
}


#[cfg(test)]
mod nipals_tests {

    use super::*;
    use rand::{thread_rng,Rng};
    use rand::distributions::Standard;
    use crate::iris_array;
    // use test::Bencher;


    #[test]
    fn iris_projection() {
        let iris = iris_array();
        let iris_m = &iris - 6.;
        println!("{:?}",iris);
        let projection = Projector::from(iris).calculate_n_projections(4);
        println!("{:?}",projection);
        panic!();
    }
}
