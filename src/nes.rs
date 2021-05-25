use std::cmp;

use ndarray::{Array, Axis, Dim, ViewRepr, concatenate};
use partial_min_max::max;
use rayon::prelude::*;

use crate::utils::{argsort, random_gaussian_matrix, random_gaussian_vector, reorder_array};
use crate::evaluator::Function;


pub struct NES<T: Function + Clone + Sync> {
    callable: T,
    mu: Array<f32, Dim<[usize; 1]>>,
    sigma: Array<f32, Dim<[usize; 1]>>,
    population_size: usize,
    learning_rate_mu: f32, 
    learning_rate_sigma: f32,
    fitness_shaping: bool,
}


impl<T: Function + Clone + Sync> NES<T> {
    pub fn new(
        callable: T, 
        mu: Array<f32, Dim<[usize; 1]>>,
        sigma: Array<f32, Dim<[usize; 1]>>,
        population_size: usize,
        learning_rate_mu: f32,
        learning_rate_sigma: f32,
        fitness_shaping: bool) -> NES<T> {
        NES {
            callable,
            mu,
            sigma,
            population_size,
            learning_rate_mu,
            learning_rate_sigma,
            fitness_shaping,
        }
    }

    fn utility_function(&self) -> Array<f32, Dim<[usize; 1]>> {
        let mut utility: Vec<f32> = Vec::new();
        let mut u: f32;
        for k in 0..self.population_size {
            u = max(0., (self.population_size as f32/2. + 1.).ln()) - (k as f32 + 1.).ln();
            utility.push(u);
        }

        utility = utility
            .iter()
            .map(|u| *u / utility.iter().sum::<f32>() - 1./self.population_size as f32)
            .collect();

        Array::from(utility)
    }

    pub fn step(&mut self) -> Array<f32, Dim<[usize; 1]>> {
        let noise: Vec<Array<f32, Dim<[usize; 1]>>> = (0..self.population_size)
            .map(|_| random_gaussian_vector(self.mu.len(), 0., 0.))
            .collect();

        let scaled_noise: Vec<Array<f32, Dim<[usize; 1]>>> = noise
            .iter()
            .map(|x| x * &self.sigma)
            .collect();

        let mut population: Vec<Array<f32, Dim<[usize; 1]>>> = scaled_noise
            .iter()
            .map(|x| &self.mu + x)
            .collect();

        let fitness: Vec<f32> = population
            .par_iter_mut()
            .map(|x| self.callable.clone().call(x.clone()))
            .collect();

        let fitness = Array::from(fitness);
        let order = argsort(&fitness);
        let ordered_fitness = reorder_array(&fitness, &order);

        fitness

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
        let mu = Array::from(vec![1., 1.]);
        let sigma = Array::from(vec![1., 1.]);
        let nes = NES::new(Evaluator::new(), mu, sigma, 10, 0.1, 0.1, false);
    }

    #[test]
    fn test_step() {
        let mu = Array::from(vec![1., 1.]);
        let sigma = Array::from(vec![1., 1.]);
        let mut nes = NES::new(Evaluator::new(), mu, sigma, 10, 0.1, 0.1, false);

        let fitness = nes.step();

        assert_eq!(fitness.len(), 10);
    }
}