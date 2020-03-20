use ndarray::prelude::*;
use crate::rank_vector::{FeatureVector,SegmentedVector,LinkedVector};
use crate::{SampleKey,SampleValue,DrawOrder};

// trait RankMatrix {
//     type Value: SampleValue;
//     type Vector: FeatureVector<V=Self::Value>;
//
//     fn array(&self) -> &Array2<Self::Value>;
//     fn reduced_array(&self) -> Array2<Self::Value>;
// }
