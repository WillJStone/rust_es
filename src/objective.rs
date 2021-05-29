use ndarray::{Array, Dim};


pub trait Objective {
    fn call(&self, x: &Array<f32, Dim<[usize; 1]>>) -> f32;
}