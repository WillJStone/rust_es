use rand;

use ndarray::{Array, Dim};
use ndarray_rand::{rand_distr::Distribution};
use ndarray_rand::rand_distr::{Normal};


pub fn random_gaussian_vector(shape: usize, mu: f32, sigma: f32) -> Array<f32, Dim<[usize; 1]>> {
    let mut rng = rand::thread_rng();
    let distribution = Normal::new(mu, sigma).unwrap();
    let vec: Vec<f32> = distribution
        .sample_iter(&mut rng)
        .take(shape)
        .collect();

    Array::from(vec)
}


pub fn argsort<T: Ord>(v: &Array<T, Dim<[usize; 1]>>) -> Array<usize, Dim<[usize; 1]>> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    
    Array::from(idx)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_random_gaussian_vector() {
        let rand_vec = random_gaussian_vector(10, 0., 1.);
        assert_eq!(rand_vec.len(), 10);
    }
}