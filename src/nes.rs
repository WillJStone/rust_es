use ndarray::{Array, Dim};

use crate::utils::random_gaussian_vector;


pub struct Evaluator {

}

impl Evaluator {
    pub fn new() -> Evaluator {
        Evaluator {}
    }
}


pub struct NES {
    callable: Evaluator,
    mu: Array<f32, Dim<[usize; 1]>>,
    sigma: f32,
    population_size: usize,
    learning_rate_mu: f32, 
    learning_rate_sigma: f32,
}


impl NES {
    pub fn new(
        callable: Evaluator, 
        mu: Array<f32, Dim<[usize; 1]>>,
        sigma: f32,
        population_size: usize,
        learning_rate_mu: f32,
        learning_rate_sigma: f32) -> NES {
        NES {
            callable,
            mu,
            sigma,
            population_size,
            learning_rate_mu,
            learning_rate_sigma,
        }
    }

    pub fn step(&mut self) {
        let mut population: Vec<Array<f32, Dim<[usize; 1]>>> = Vec::new();
        for i in 0..self.population_size { 
            let noise = random_gaussian_vector(self.mu.len(), 0., 1.);
            let z = self.mu.clone() + noise * self.sigma;
            population.push(z);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nes_new() {
        let mu = random_gaussian_vector(2, 0., 1.);
        let nes = NES::new(Evaluator::new(), mu, 0.9, 10, 0.1, 0.1);
    }
}