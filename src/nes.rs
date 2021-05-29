use ndarray::{Array, Dim};
use partial_min_max::max;
use rayon::prelude::*;

use crate::utils::{argsort, array_from_vec_of_arrays, random_gaussian_vector, reorder_vec};
use crate::objective::Objective;


pub struct NES<T: Objective + Clone + Sync> {
    callable: T,
    pub mu: Array<f32, Dim<[usize; 1]>>,
    sigma: Array<f32, Dim<[usize; 1]>>,
    population_size: usize,
    learning_rate_mu: f32, 
    learning_rate_sigma: f32,
    fitness_shaping: bool,
}


impl<T: Objective + Clone + Sync> NES<T> {
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

    pub fn step(&mut self) {
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
        
        let mut noise_squared = noise.clone();
        noise_squared.map_inplace(|x| *x = x.powi(2) - 1.);
        let d = utility.dot(&noise_squared);
        let delta_sigma = self.learning_rate_sigma / 2. * &self.sigma * 1./self.population_size as f32  * d ;
        self.sigma = &self.sigma + delta_sigma;
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
    
    
    impl Objective for Evaluator {
        fn call(&self, _x: &Array<f32, Dim<[usize; 1]>>) -> f32 {
            42.
        }
    }

    #[test]
    fn test_nes_new() {
        let mu = Array::from(vec![1., 1.]);
        let sigma = Array::from(vec![1., 1.]);
        let _nes = NES::new(Evaluator::new(), mu, sigma, 10, 0.1, 0.1, false);
    }

    #[test]
    fn test_step() {
        let mu = Array::from(vec![1., 1.]);
        let sigma = Array::from(vec![1., 1.]);
        let mut nes = NES::new(Evaluator::new(), mu, sigma, 10, 0.1, 0.1, false);

        nes.step();
    }
}