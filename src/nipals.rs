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

    pub fn calculate_n_projections(mut self,n:usize) -> (Array2<f64>,Array2<f64>,Array1<f64>,Array1<f64>) {
        let mut loadings = Array2::zeros((n,self.converted.dim().0));
        let mut scores = Array2::zeros((n,self.converted.dim().1));
        for i in 0..n {
            println!("Projection {:?}",i);
            println!("{:?}",self);
            let (n_loadings,n_scores,_,_)= self.calculate_projection();
            loadings.row_mut(i).assign(&n_loadings);
            scores.row_mut(i).assign(&n_scores);
        }
        (loadings,scores,self.means,self.scale_factors)
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
    // use test::Bencher;

    fn iris_array() -> Array2<f64> {
        array![[5.1,3.5,1.4,0.2],
        [4.9,3.0,1.4,0.2],
        [4.7,3.2,1.3,0.2],
        [4.6,3.1,1.5,0.2],
        [5.0,3.6,1.4,0.2],
        [5.4,3.9,1.7,0.4],
        [4.6,3.4,1.4,0.3],
        [5.0,3.4,1.5,0.2],
        [4.4,2.9,1.4,0.2],
        [4.9,3.1,1.5,0.1],
        [5.4,3.7,1.5,0.2],
        [4.8,3.4,1.6,0.2],
        [4.8,3.0,1.4,0.1],
        [4.3,3.0,1.1,0.1],
        [5.8,4.0,1.2,0.2],
        [5.7,4.4,1.5,0.4],
        [5.4,3.9,1.3,0.4],
        [5.1,3.5,1.4,0.3],
        [5.7,3.8,1.7,0.3],
        [5.1,3.8,1.5,0.3],
        [5.4,3.4,1.7,0.2],
        [5.1,3.7,1.5,0.4],
        [4.6,3.6,1.0,0.2],
        [5.1,3.3,1.7,0.5],
        [4.8,3.4,1.9,0.2],
        [5.0,3.0,1.6,0.2],
        [5.0,3.4,1.6,0.4],
        [5.2,3.5,1.5,0.2],
        [5.2,3.4,1.4,0.2],
        [4.7,3.2,1.6,0.2],
        [4.8,3.1,1.6,0.2],
        [5.4,3.4,1.5,0.4],
        [5.2,4.1,1.5,0.1],
        [5.5,4.2,1.4,0.2],
        [4.9,3.1,1.5,0.1],
        [5.0,3.2,1.2,0.2],
        [5.5,3.5,1.3,0.2],
        [4.9,3.1,1.5,0.1],
        [4.4,3.0,1.3,0.2],
        [5.1,3.4,1.5,0.2],
        [5.0,3.5,1.3,0.3],
        [4.5,2.3,1.3,0.3],
        [4.4,3.2,1.3,0.2],
        [5.0,3.5,1.6,0.6],
        [5.1,3.8,1.9,0.4],
        [4.8,3.0,1.4,0.3],
        [5.1,3.8,1.6,0.2],
        [4.6,3.2,1.4,0.2],
        [5.3,3.7,1.5,0.2],
        [5.0,3.3,1.4,0.2],
        [7.0,3.2,4.7,1.4],
        [6.4,3.2,4.5,1.5],
        [6.9,3.1,4.9,1.5],
        [5.5,2.3,4.0,1.3],
        [6.5,2.8,4.6,1.5],
        [5.7,2.8,4.5,1.3],
        [6.3,3.3,4.7,1.6],
        [4.9,2.4,3.3,1.0],
        [6.6,2.9,4.6,1.3],
        [5.2,2.7,3.9,1.4],
        [5.0,2.0,3.5,1.0],
        [5.9,3.0,4.2,1.5],
        [6.0,2.2,4.0,1.0],
        [6.1,2.9,4.7,1.4],
        [5.6,2.9,3.6,1.3],
        [6.7,3.1,4.4,1.4],
        [5.6,3.0,4.5,1.5],
        [5.8,2.7,4.1,1.0],
        [6.2,2.2,4.5,1.5],
        [5.6,2.5,3.9,1.1],
        [5.9,3.2,4.8,1.8],
        [6.1,2.8,4.0,1.3],
        [6.3,2.5,4.9,1.5],
        [6.1,2.8,4.7,1.2],
        [6.4,2.9,4.3,1.3],
        [6.6,3.0,4.4,1.4],
        [6.8,2.8,4.8,1.4],
        [6.7,3.0,5.0,1.7],
        [6.0,2.9,4.5,1.5],
        [5.7,2.6,3.5,1.0],
        [5.5,2.4,3.8,1.1],
        [5.5,2.4,3.7,1.0],
        [5.8,2.7,3.9,1.2],
        [6.0,2.7,5.1,1.6],
        [5.4,3.0,4.5,1.5],
        [6.0,3.4,4.5,1.6],
        [6.7,3.1,4.7,1.5],
        [6.3,2.3,4.4,1.3],
        [5.6,3.0,4.1,1.3],
        [5.5,2.5,4.0,1.3],
        [5.5,2.6,4.4,1.2],
        [6.1,3.0,4.6,1.4],
        [5.8,2.6,4.0,1.2],
        [5.0,2.3,3.3,1.0],
        [5.6,2.7,4.2,1.3],
        [5.7,3.0,4.2,1.2],
        [5.7,2.9,4.2,1.3],
        [6.2,2.9,4.3,1.3],
        [5.1,2.5,3.0,1.1],
        [5.7,2.8,4.1,1.3],
        [6.3,3.3,6.0,2.5],
        [5.8,2.7,5.1,1.9],
        [7.1,3.0,5.9,2.1],
        [6.3,2.9,5.6,1.8],
        [6.5,3.0,5.8,2.2],
        [7.6,3.0,6.6,2.1],
        [4.9,2.5,4.5,1.7],
        [7.3,2.9,6.3,1.8],
        [6.7,2.5,5.8,1.8],
        [7.2,3.6,6.1,2.5],
        [6.5,3.2,5.1,2.0],
        [6.4,2.7,5.3,1.9],
        [6.8,3.0,5.5,2.1],
        [5.7,2.5,5.0,2.0],
        [5.8,2.8,5.1,2.4],
        [6.4,3.2,5.3,2.3],
        [6.5,3.0,5.5,1.8],
        [7.7,3.8,6.7,2.2],
        [7.7,2.6,6.9,2.3],
        [6.0,2.2,5.0,1.5],
        [6.9,3.2,5.7,2.3],
        [5.6,2.8,4.9,2.0],
        [7.7,2.8,6.7,2.0],
        [6.3,2.7,4.9,1.8],
        [6.7,3.3,5.7,2.1],
        [7.2,3.2,6.0,1.8],
        [6.2,2.8,4.8,1.8],
        [6.1,3.0,4.9,1.8],
        [6.4,2.8,5.6,2.1],
        [7.2,3.0,5.8,1.6],
        [7.4,2.8,6.1,1.9],
        [7.9,3.8,6.4,2.0],
        [6.4,2.8,5.6,2.2],
        [6.3,2.8,5.1,1.5],
        [6.1,2.6,5.6,1.4],
        [7.7,3.0,6.1,2.3],
        [6.3,3.4,5.6,2.4],
        [6.4,3.1,5.5,1.8],
        [6.0,3.0,4.8,1.8],
        [6.9,3.1,5.4,2.1],
        [6.7,3.1,5.6,2.4],
        [6.9,3.1,5.1,2.3],
        [5.8,2.7,5.1,1.9],
        [6.8,3.2,5.9,2.3],
        [6.7,3.3,5.7,2.5],
        [6.7,3.0,5.2,2.3],
        [6.3,2.5,5.0,1.9],
        [6.5,3.0,5.2,2.0],
        [6.2,3.4,5.4,2.3],
        [5.9,3.0,5.1,1.8]]
    }

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
