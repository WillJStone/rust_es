pub extern crate ndarray;
pub extern crate ndarray_rand;

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
    fn call(&self, x: Array<f32, Dim<[usize; 1]>>) -> f32 {
        x.dot(&x.clone().t())
    }
}


fn main() {
    let mu = Array::from(vec![4., 1.]);
    let sigma = Array::from(vec![0., 0.1]);
    let mut nes = NES::new(
        Quadratic::new(), 
        mu, 
        sigma, 
        10, 
        0.9, 
        0.1,
        false,
    );

    let fitness = nes.step();

    println!("{:?}", fitness);
}
