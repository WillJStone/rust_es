use ndarray::{Array, Dim};
use rayon::prelude::*;

use crate::utils::{argsort, random_gaussian_vector};
use crate::evaluator::Function;


pub struct NES<T: Function + Clone + Sync> {
    callable: T,
    mu: Array<f32, Dim<[usize; 1]>>,
    sigma: Array<f32, Dim<[usize; 1]>>,
    population_size: usize,
    learning_rate_mu: f32, 
    learning_rate_sigma: f32,
}


impl<T: Function + Clone + Sync> NES<T> {
    pub fn new(
        callable: T, 
        mu: Array<f32, Dim<[usize; 1]>>,
        sigma: Array<f32, Dim<[usize; 1]>>,
        population_size: usize,
        learning_rate_mu: f32,
        learning_rate_sigma: f32) -> NES<T> {
        NES {
            callable,
            mu,
            sigma,
            population_size,
            learning_rate_mu,
            learning_rate_sigma,
        }
    }

    pub fn step(&mut self) -> Array<f32, Dim<[usize; 1]>> {
        let mut population: Vec<Array<f32, Dim<[usize; 1]>>> = Vec::new();
        for _ in 0..self.population_size { 
            let noise = random_gaussian_vector(self.mu.len(), 0., 1.);
            let scaled_noise: Array<f32, Dim<[usize; 1]>> = self.sigma.clone() * noise;
            let z = self.mu.clone() + scaled_noise;
            population.push(z);
        }

        let fitness: Vec<f32> = population
            .par_iter_mut()
            .map(|x| self.callable.clone().call(x.clone()))
            .collect();

        Array::from(fitness)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone)]
    pub struct Evaluator {

    }
    
    impl Evaluator {
        pub fn new() -> Evaluator {
            Evaluator {}
        }
    }
    
    
    impl Function for Evaluator {
        fn call(&self, x: Array<f32, Dim<[usize; 1]>>) -> f32 {
            42.
        }
    }

    #[test]
    fn test_nes_new() {
        let mu = random_gaussian_vector(2, 0., 1.);
        let sigma = Array::from(vec![1., 1.]);
        let nes = NES::new(Evaluator::new(), mu, sigma, 10, 0.1, 0.1);
    }

    #[test]
    fn test_step() {
        let mu = random_gaussian_vector(2, 0., 1.);
        let sigma = Array::from(vec![1., 1.]);
        let mut nes = NES::new(Evaluator::new(), mu, sigma, 10, 0.1, 0.1);

        let fitness = nes.step();

        assert_eq!(fitness.len(), 10);
    }
}