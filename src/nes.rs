use std::cmp;

use ndarray::{Array, Axis, Dim, ViewRepr, concatenate};
use partial_min_max::max;
use rayon::prelude::*;

use crate::utils::{self, argsort, array_from_vec_of_arrays, random_gaussian_matrix, random_gaussian_vector, reorder_array, reorder_vec};
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
        let mut noise: Vec<Array<f32, Dim<[usize; 1]>>> = (0..self.population_size)
            .map(|_| random_gaussian_vector(self.mu.len(), 0., 1.))
            .collect();

        let scaled_noise: Vec<Array<f32, Dim<[usize; 1]>>> = noise
            .iter()
            .map(|x| x * &self.sigma)
            .collect();

        let population: Vec<Array<f32, Dim<[usize; 1]>>> = scaled_noise
            .iter()
            .map(|x| &self.mu + x)
            .collect();

        let fitness: Vec<f32> = population
            .par_iter()
            .map(|x| self.callable.clone().call(x))
            .collect();

        let fitness = Array::from(fitness);
        let utility: Array<f32, Dim<[usize; 1]>>;
        if self.fitness_shaping {
            let order = argsort(&fitness);
            noise = reorder_vec(&noise, order.clone());
            utility = self.utility_function();
        } else {
            utility = fitness;
        }

        let noise = array_from_vec_of_arrays(noise);
        let d = utility.dot(&noise);
        let delta_mu = &self.sigma * self.learning_rate_mu * d * 1./self.population_size as f32;
        self.mu = &self.mu + delta_mu;
        
        
        // let delta_mu: Array<f32, Dim<[usize; 1]>> = self.sigma
        //     .iter()
        //     .map(|x| x * self.learning_rate_mu * 1./self.population_size as f32 * utility.dot(&noise))
        //     .collect();
        // //let fitness = Array::from(utility);

        // utility
        self.mu.clone()

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
        fn call(&self, x: &Array<f32, Dim<[usize; 1]>>) -> f32 {
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