pub extern crate ndarray;
pub extern crate ndarray_rand;
pub extern crate ndarray_parallel;

use ndarray::{Array, Dim};

use crate::objective::Objective;
use crate::nes::NES;

mod nes;
mod utils;
mod objective;

#[derive(Clone)]
pub struct Quadratic {

}

impl Quadratic {
    pub fn new() -> Quadratic {
        Quadratic {}
    }
}


impl Objective for Quadratic {
    fn call(&self, x: &Array<f32, Dim<[usize; 1]>>) -> f32 {
        x.dot(&x.clone().t())
    }
}


fn main() {
    let mu = Array::from(vec![4., 8.]);
    let sigma = Array::from(vec![1., 1.]);
    let callable = Quadratic::new();
    let mut nes = NES::new(
        callable.clone(), 
        mu, 
        sigma, 
        10, 
        1.5, 
        0.1,
        true,
    );

    for i in 0..5000 {
        nes.step();
    }

    
}
