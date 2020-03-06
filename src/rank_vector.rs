use std::f64;
use std::fmt;
use std::hash::Hash;
use std::cmp::Eq;
use num_traits::{Num,Zero,PrimInt,FromPrimitive,ToPrimitive,NumCast,Pow,};
use std::convert::{From};
use std::marker::PhantomData;
use std::collections::{HashSet,HashMap};
use std::cmp::Ordering;
use smallvec::SmallVec;
use std::ops::{Index,IndexMut,SubAssign, AddAssign,Mul};
use std::fmt::Debug;
use std::clone::Clone;
use std::borrow::{Borrow,BorrowMut};
use std::iter::{Take,Skip,FlatMap,Chain};
use crate::io::DropMode;
use crate::{SampleKey,SampleValue};


#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
pub struct Node<K,V>
where
    K: SampleKey,
    V: SampleValue,
{
    value: V,
    squared_value:V,
    key: K,
    previous: K,
    next: K,
    segment: usize,
}

impl<K,V> Node<K,V>
where
    K: SampleKey,
    V: SampleValue,
{

    fn blank(key:K) -> Node<K,V> {
        Node {
            value:V::zero(),
            squared_value:V::zero(),
            key: key,
            previous:key,
            next:key,
            segment:0,
        }
    }

}

pub trait FeatureVector
{
    type K: SampleKey;
    type V: SampleValue;

    fn central_tendency(&self) -> Self::V;
    fn dispersion(&self) -> Self::V;
}


pub trait LinkedVector: Sized
{
    type K: SampleKey;
    type V: SampleValue;
    type Arena: Index<Self::K,Output=Node<Self::K,Self::V>> + IndexMut<Self::K,Output=Node<Self::K,Self::V>> + Debug;


    fn arena(&self) -> &Self::Arena;
    fn arena_mut(&mut self) -> &mut Self::Arena;

    fn unlink_node(&mut self,target_key:Self::K) -> Node<Self::K,Self::V> {
        let target = self.arena()[target_key];
        self.link_nodes(target.previous, target.next);
        target
    }

    fn link_nodes(&mut self, left:Self::K,right:Self::K) {
        {self.arena_mut()[left].next = right}
        {self.arena_mut()[right].previous = left}
        let seg = self.arena()[left].segment;
        {self.arena_mut()[right].segment = seg}

    }

    fn insert_node(&mut self, previous_key:Self::K, mut target: Node<Self::K,Self::V>, next_key:Self::K) {
        target.previous = previous_key;
        target.next = next_key;
        target.segment = self.arena()[previous_key].segment;
        let target_key = target.key;
        {self.arena_mut()[previous_key].next = target_key}
        {self.arena_mut()[next_key].previous = target_key}
        {self.arena_mut()[target_key] = target}
    }

    fn left_crawler(&self,start:Self::K) -> LeftCrawler<Self,Self::K,Self::V> {
        LeftCrawler{vector:&self,key:start,phantom:PhantomData}
    }

    fn right_crawler(&self,start:Self::K) -> RightCrawler<Self,Self::K,Self::V> {
        RightCrawler{vector:&self,key:start,phantom:PhantomData}
    }


}

pub trait SegmentedVector: LinkedVector + Sized
{

    type V: SampleValue

    fn len(&self) -> usize;
    fn balance(&mut self);
    fn segments(&self) -> &[IndexSegment<V>];
    fn segments_mut(&mut self) -> &mut [IndexSegment<V>];

    fn crawl_segment_reverse(&self,segment:usize) -> Take<Skip<LeftCrawler<Self,usize,V>>> {
        self.left_crawler(self.segments()[segment].right).skip(1).take(self.segments()[segment].len())
    }

    fn crawl_segment_ordered(&self,segment:usize) -> Take<Skip<RightCrawler<Self,usize,V>>> {
        self.right_crawler(self.segments()[segment].left).skip(1).take(self.segments()[segment].len())
    }

    fn ordered_values(&self) -> Vec<V>{
        (0..self.segments().len()).flat_map(|s| self.crawl_segment_ordered(s)).map(|n| n.value).collect()
    }

    fn ordered_indices(&self) -> Vec<usize>{
        (0..self.segments().len()).flat_map(|s| self.crawl_segment_ordered(s)).map(|n| n.key).collect()
    }

    fn pop(&mut self,key:usize) {
        let target = LinkedVector::unlink_node(self,key);
        let segment = &mut self.segments_mut()[target.segment];
        segment.pop(&target);
        self.balance();
    }

    fn push_segment_left(&mut self,segment:usize,mut node:Node<usize,V>) {
        let left_cap = {self.segments()[segment].left};
        let left_edge = {self.arena()[left_cap].next};
        self.segments_mut()[segment].push(&node);
        self.insert_node(left_cap, node, left_edge)
    }

    fn pop_segment_left(&mut self,segment:usize) -> Node<usize,V> {
        let left_cap = {self.segments()[segment].left};
        let left_edge = {self.arena()[left_cap].next};
        let node = self.unlink_node(left_edge);
        self.segments_mut()[segment].pop(&node);
        node
    }

    fn push_segment_right(&mut self,segment:usize,mut node:Node<usize,V>) {
        let right_cap = {self.segments()[segment].right};
        let right_edge = {self.arena()[right_cap].previous};
        self.segments_mut()[segment].push(&node);
        self.insert_node(right_edge, node, right_cap)
    }

    fn pop_segment_right(&mut self,segment:usize) -> Node<usize,V> {
        let right_cap = {self.segments()[segment].right};
        let right_edge = {self.arena()[right_cap].previous};
        let node = self.unlink_node(right_edge);
        self.segments_mut()[segment].pop(&node);
        node
    }

    fn shift_boundary_left(&mut self,segment_left:usize,segment_right:usize) {
        let mut target = self.pop_segment_right(segment_left);
        self.push_segment_left(segment_right, target);
        println!("shifted bound left {:?}",(segment_left,segment_right));
        println!("{:?}",self.segments());
    }

    fn shift_boundary_right(&mut self,segment_left:usize,segment_right:usize) {
        let mut target = self.pop_segment_left(segment_right);
        self.push_segment_right(segment_left, target);
        println!("shifted bound left {:?}",(segment_left,segment_right));
        println!("{:?}",self.segments());
    }

    fn initialize(&mut self,len:usize) {
        let seg_len = {self.segments().len()};
        for i in 0..seg_len {
            // println!("Initializing segment {:?}",i);
            // println!("{:?},{:?}",len + (i*2),len + (i*2) + 1);
            let (seg,(e1,e2)) = IndexSegment::offset_segment(len,i);
            {self.segments_mut()[i] = seg;}
            {self.arena_mut()[len + (i*2)] = e1;}
            {self.arena_mut()[len + (i*2) + 1] = e2;}
        }
    }


    fn link(&mut self, sorted_input:&[(usize,V)]) -> &mut Self {
        let input_len = sorted_input.len();
        let mut previous_key = self.segments()[0].left();
        for ((key,value),(next_key,_)) in sorted_input.iter().zip(sorted_input.iter().skip(1)) {
            let node = Node {
                next:*next_key,
                previous: previous_key,
                value: *value,
                squared_value: value.pow(2),
                key: *key,
                segment: 0,
            };
            {self.segments_mut()[0].push(&node)}
            {self.arena_mut()[previous_key].next = *key}
            {self.arena_mut()[*key] = node};
            previous_key = *key;
            println!("finished key {:?}",previous_key);
        }
        if let Some((final_key,final_value)) = sorted_input.last() {
            let cap_key = self.segments()[0].right;
            let node = Node {
                next:cap_key,
                previous: previous_key,
                value: *final_value,
                squared_value: final_value.pow(2),
                key: *final_key,
                segment: 0,
            };
            {self.segments_mut()[0].push(&node)}
            {self.arena_mut()[*final_key] = node};
            {self.arena_mut()[cap_key].previous = *final_key}
        }
        self.balance();
        println!("Linking almost done");
        println!("{:?}",self.arena());
        println!("{:?}",self.segments());
        self
    }

}

pub struct RightCrawler<'v,SV,K,V>
where
    SV:LinkedVector<K,V>,
    K: SampleKey,
    V: SampleValue,
 {
    vector: &'v SV,
    key: K,
    phantom: PhantomData<(&'v V,&'v K)>
}

impl<'v,SV,K,V> Iterator for RightCrawler<'v,SV,K,V>
where
    SV:LinkedVector<K,V>,
    K: SampleKey,
    V: SampleValue,
{
    type Item = &'v Node<K,V>;

    fn next(&mut self) -> Option<&'v Node<K,V>> {
        println!("Crawling right:{:?}",self.key);
        let node = &self.vector.arena()[self.key];
        self.key = node.next;
        return Some(node)
    }
}


pub struct LeftCrawler<'v,SV,K,V>
where
    SV:LinkedVector<K,V>,
    K: SampleKey,
    V: SampleValue,
 {
    vector: &'v SV,
    key: K,
    phantom: PhantomData<(&'v V,&'v K)>
}

impl<'v,SV,K,V> Iterator for LeftCrawler<'v,SV,K,V>
where
    SV:LinkedVector<K,V>,
    K: SampleKey,
    V: SampleValue,
{
    type Item = &'v Node<K,V>;

    fn next(&mut self) -> Option<&'v Node<K,V>> {

        let node = &self.vector.arena()[self.key];
        self.key = node.previous;
        return Some(node)
    }
}


trait Segment<K,V>
where
    K: SampleKey,
    V: SampleValue,
{

    fn left(&self) -> K;
    fn right(&self) -> K;
    fn len(&self) -> usize;
    fn sum(&self) -> V;
    fn squared_sum(&self) -> V;

    fn len_mut(&mut self) -> &mut usize;
    fn sum_mut(&mut self) -> &mut V;
    fn squared_sum_mut(&mut self) -> &mut V;

    fn pop(&mut self,node:&Node<K,V>) {
        println!("Popping {:?}", node);
        *self.sum_mut() -= node.value;
        *self.squared_sum_mut() -= node.squared_value;
        *self.len_mut() -= 1;
        println!("Done {:?}",(self.sum(),self.len()));
    }
    fn push(&mut self, node:&Node<K,V>) {
        println!("Pushing {:?}", node);
        *self.sum_mut() += node.value;
        *self.squared_sum_mut() += node.squared_value;
        *self.len_mut() += 1;
        println!("Done {:?}",(self.sum(),self.len()));
    }
}

#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
struct IndexSegment<V: SampleValue> {
    left:usize,
    right:usize,
    sum:V,
    squared_sum:V,
    len:usize,
}

impl<V: SampleValue> Segment<usize,V> for IndexSegment<V> {
    fn left(&self) -> usize {
        self.left
    }
    fn right(&self) -> usize {
        self.right
    }
    fn len(&self) -> usize {
        self.len
    }
    fn sum(&self) -> V {
        self.sum
    }
    fn squared_sum(&self) -> V {
        self.squared_sum
    }

    fn len_mut(&mut self) -> &mut usize {
        &mut self.len
    }
    fn sum_mut(&mut self) -> &mut V {
        &mut self.sum
    }
    fn squared_sum_mut(&mut self) -> &mut V {
        &mut self.squared_sum
    }

}

impl<V:SampleValue> IndexSegment<V> {

    fn blank() -> IndexSegment<V> {
        IndexSegment{
            left:0,
            right:0,
            sum:V::zero(),
            squared_sum:V::zero(),
            len:0,
        }
    }

    fn offset_segment(offset:usize,segment:usize) -> (IndexSegment<V>,(Node<usize,V>,Node<usize,V>)) {
        let i1 = offset+(segment*2);
        let i2 = offset+(segment*2)+1;
        let vec_seg = IndexSegment {
            left: i1,
            right: i2,
            sum: V::zero(),
            squared_sum: V::zero(),
            len: 0,};
        let e1 = Node {
            key: i1,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment: segment,
        };
        let e2 = Node {
            key: i2,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment:segment,
        };
        (vec_seg,(e1,e2))
    }

}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct MedianArena<A,V>
where
    A: NodeArena<usize,V>,
    V: SampleValue,
{
    segments: [IndexSegment<V>;3],
    arena: A,
}

type MedianArray<V:SampleValue> = MedianArena<NodeArray<V>,V>;
type MedianVector<V:SampleValue> = MedianArena<Vec<Node<usize,V>>,V>;

impl<A,V> LinkedVector<usize,V> for MedianArena<A,V>
where
    A: NodeArena<usize,V>,
    V: SampleValue,
{
    type Arena = A;

    fn arena(&self) -> &Self::Arena {
        &self.arena
    }

    fn arena_mut(&mut self) -> &mut Self::Arena {
        &mut self.arena
    }
}


impl<A,V> SegmentedVector<V> for MedianArena<A,V>
where
    A: NodeArena<usize,V>,
    V: SampleValue,
{

    fn len(&self) -> usize {
        self.segments().iter().map(|s| s.len()).sum()
    }

    fn segments(&self) -> &[IndexSegment<V>] {
        &self.segments
    }
    fn segments_mut(&mut self) -> &mut [IndexSegment<V>] {
        &mut self.segments
    }

    fn balance(&mut self) {
        self.size_median();
        // println!("Balancing");
        while {self.segments()[0].len()} < {self.segments()[2].len()} {
            // println!("Shifting median left");
            self.shift_median_right();
        }
        while {self.segments()[0].len()} > {self.segments()[2].len()} {
            // println!("Shifting median right");
            self.shift_median_left();
        }
    }

}

impl<A,V> MedianArena<A,V>
where
    A: NodeArena<usize,V>,
    V: SampleValue,
{

        fn with_capacity(capacity:usize) -> MedianArena<A,V> {

            let mut mv = MedianArena{
                segments: [IndexSegment::blank();3],
                arena: NodeArena::<usize,V>::with_capacity(capacity+6),
            };
            mv.initialize(capacity);
            mv
        }

        fn size_median(&mut self) {
            // println!("Initializing median");

            if self.segments[1].len() < 1 {
                // println!("Own length less than 1");
                // println!("{:?}",self);
                if self.segments[0].len() > 0 {
                    // println!("Shifting boundary left");
                    self.shift_boundary_left(0, 1)
                }
                else if self.segments[2].len() > 0 {
                    // println!("Shifting boundary left");
                    self.shift_boundary_right(1,2)
                }
                // else {println!("No shift"); panic!()}
            }
            // println!("Initialized to at least 1");
            while self.segments[1].len > 2 {
                self.shift_boundary_left(1,2)
            }

            // println!("{:?}",self.segments());
        }

        fn link(sorted_input:&[(usize,V)]) -> Self {
            let mut mv = Self::with_capacity(sorted_input.len());
            SegmentedVector::link(&mut mv,sorted_input);
            mv.balance();
            mv
        }

        fn shift_median_left(&mut self) {
            match self.segments[1].len() {
                1 => {self.shift_boundary_left(0,1)},
                2 => {self.shift_boundary_left(1,2)},
                _ => {panic!(format!("Median de-synchronized:{:?}",self))}
            }
        }

        fn shift_median_right(&mut self) {
            match self.segments[1].len() {
                1 => {self.shift_boundary_right(1,2)},
                2 => {self.shift_boundary_right(0,1)},
                _ => {panic!(format!("Median de-synchronized:{:?}",self))}
            }
        }

        fn median(&self) -> V {
            if self.segments[1].len() > 0 {
                self.segments[1].sum / (V::from(self.segments[1].len()).expect("Cast failure"))
            }
            else {V::zero()}
        }

        fn ssme(&self) -> V {
            let median = self.median();
            let squared_sum = self.segments()[0].squared_sum + self.segments()[1].squared_sum + self.segments()[2].squared_sum;
            let sum = self.segments()[0].sum + self.segments()[1].sum + self.segments()[2].sum;
            squared_sum - ((V::from(2).expect("Cast failure"))*median*sum) + ((V::from(self.len()).expect("Cast failure")) * (median.pow(2)))
        }

}


pub trait NodeArena<K:SampleKey,V:SampleValue>: Index<usize,Output=Node<usize,V>> + IndexMut<usize,Output=Node<usize,V>> + Debug + Clone {
    const ARRAY_LIMIT:usize = 1024;
    fn with_capacity(capacity:usize) -> Self;
}

impl<V: SampleValue> NodeArena<usize,V> for Vec<Node<usize,V>>{
    fn with_capacity(capacity:usize) -> Vec<Node<usize,V>> {
        vec![Node::blank(0);capacity]
    }
}
impl<V: SampleValue> NodeArena<usize,V> for NodeArray<V>{
    fn with_capacity(capacity:usize) -> NodeArray<V> {
        if capacity <= 1018 { NodeArray([Node::blank(0);1024])}
        else {panic!("Exceeded capacity of array: {:?}",capacity)}
    }

}

#[derive(Clone)]
pub struct NodeArray<V:SampleValue>([Node<usize,V>;1024]);

impl<V> Index<usize> for NodeArray<V>
where
    V: SampleValue,
{
    type Output = Node<usize,V>;
    fn index(&self,index:usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V> IndexMut<usize> for NodeArray<V>
where
    V: SampleValue,
{
    fn index_mut(&mut self,index:usize) -> &mut Node<usize,V> {
        &mut self.0[index]
    }
}

impl<V:SampleValue> Debug for NodeArray<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}",self.0.to_vec())
    }
}


#[cfg(test)]
mod random_forest_tests {

    use rand::prelude::*;
    use super::*;


    fn slow_median(values: &Vec<f64>) -> f64 {
        let median: f64;
        if values.len() < 1 {
            return 0.
        }

        if values.len()%2==0 {
            median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
        }
        else {
            median = values[(values.len()-1)/2];
        }

        median

    }

    fn slow_mad(values: &Vec<f64>) -> f64 {
        let median: f64;
        if values.len() < 1 {
            return 0.
        }
        if values.len()%2==0 {
            median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
        }
        else {
            median = values[(values.len()-1)/2];
        }

        let mut abs_deviations: Vec<f64> = values.iter().map(|x| (x-median).abs()).collect();

        abs_deviations.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

        let mad: f64;
        if abs_deviations.len()%2==0 {
            mad = (abs_deviations[abs_deviations.len()/2] + abs_deviations[abs_deviations.len()/2 - 1]) as f64 / 2.;
        }
        else {
            mad = abs_deviations[(abs_deviations.len()-1)/2];
        }

        mad

    }

    fn slow_ssme(values: &Vec<f64>) -> f64 {
        let median = slow_median(values);
        values.iter().map(|x| (x - median).powi(2)).sum()
    }

    fn slow_sme(values: &Vec<f64>) -> f64 {
        let median = slow_median(values);
        values.iter().map(|x| (x - median).abs()).sum()
    }

    fn simple_values() -> Vec<f64> {
        vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]
    }

    fn random_draw_order() -> Vec<usize> {
        let mut d = (0..1000).collect::<Vec<usize>>();
        &mut d[..].shuffle(&mut thread_rng());
        d
    }

    fn random_floats() -> Vec<f64> {
        (0..1000).map(|_| thread_rng().gen::<f64>()).collect()
    }


    fn argsorted() -> Vec<(usize,f64)> {
        let mut s = simple_values();
        let mut paired: Vec<(usize,f64)> = s.into_iter().enumerate().collect();
        paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        return paired
    }

    #[test]
    fn ordered_value_test() {
        let argsorted = argsorted();
        let ordered_values = vec![-3.,-2.,-1.,0.,5.,10.,15.,20.];
        println!("{:?}",ordered_values);
        let mut mv = MedianVector::<f64>::with_capacity(8);
        mv.link(&argsorted);
        println!("{:?}",ordered_values);
        assert_eq!(ordered_values,mv.ordered_values());
    }

    #[test]
    fn median_test() {
        let mut mv = MedianVector::<f64>::with_capacity(8);
        mv.link(&argsorted());
        println!("{:?}",mv);
        println!("{:?}",mv.median());
        assert_eq!(2.5,mv.median());
        mv.pop(0);
        println!("{:?}",mv);
        assert_eq!(0.,mv.median());

    }

    #[test]
    fn random_median_test() {

        let floats = random_floats();
        let draw_order = random_draw_order();
        let mut argsorted: Vec<(usize,f64)> = floats.into_iter().enumerate().collect();
        argsorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut mv = MedianVector::link(&argsorted);
        for i in draw_order {
            mv.pop(i);
            let ordered_values = mv.ordered_values();
            if (slow_median(&ordered_values)-mv.median()).abs() > 0.000001 {
                println!("{:?}",(slow_median(&ordered_values),mv.median()));
                println!("{:?}",(slow_median(&ordered_values)-mv.median()).abs());
                panic!();
            }
            if (slow_ssme(&ordered_values)-mv.ssme()).abs() > 0.000001 {
                println!("{:?}",(slow_ssme(&ordered_values),mv.ssme()));
                println!("{:?}",(slow_ssme(&ordered_values)-mv.ssme()).abs());
                panic!();
            }

        }
    }


}



// #[derive(Clone,Debug,Serialize,Deserialize)]
// pub struct ArrayArena<V>
// where
//     V: SampleValue,
// {
//     array:[Node<usize,V>;1024],
// }
//
// impl<V> Index<usize> for ArrayArena<V>
// where
//     V: SampleValue,
// {
//     type Output = Node<usize,V>;
//     fn index(&self,index:usize) -> Node<usize,V> {
//         self.array[index]
//     }
// }
//
// impl<V> NodeArena<usize,V> for ArrayArena<V>
// where
//     V: SampleValue,
// {
//     fn with_capacity(template:Node<>)
// }
//
// #[derive(Clone,Debug,Serialize,Deserialize)]
// pub struct MedianVector<A,V>
// where
//     V: SampleValue,
//     A: NodeArena<usize,V>,
// {
//     segments: [IndexSegment<V>;3],
//     arena: A,
// }
//
//
