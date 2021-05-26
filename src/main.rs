pub extern crate ndarray;
pub extern crate ndarray_rand;
pub extern crate ndarray_parallel;

use ndarray::{Array, Dim};

use crate::evaluator::Function;
use crate::nes::NES;

mod nes;
mod utils;
mod evaluator;

#[derive(Clone)]
pub struct Quadratic {

}

impl Quadratic {
    pub fn new() -> Quadratic {
        Quadratic {}
    }
}


impl Function for Quadratic {
    fn call(&self, x: &Array<f32, Dim<[usize; 1]>>) -> f32 {
        x.dot(&x.clone().t())
    }
}


fn main() {
    let mu = Array::from(vec![4., 8.]);
    let sigma = Array::from(vec![1., 1.]);
    let mut nes = NES::new(
        Quadratic::new(), 
        mu, 
        sigma, 
        10, 
        0.9, 
        0.1,
        true,
    );

    for i in 0..500 {
        nes.step();
        println!("mu at step {}: {:?}", i, nes.mu);
    }

    
}
