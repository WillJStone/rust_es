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


pub fn random_gaussian_matrix(shape: (usize, usize), mu: f32, sigma: f32) -> Array<f32, Dim<[usize; 2]>> {
    let mut rng = rand::thread_rng();
    let distribution = Normal::new(mu, sigma).unwrap();
    let vec: Vec<f32> = distribution
        .sample_iter(&mut rng)
        .take(shape.0 * shape.1)
        .collect();

    Array::from_shape_vec(shape, vec).unwrap()
}


pub fn argsort<T: PartialOrd>(v: &Array<T, Dim<[usize; 1]>>) -> Array<usize, Dim<[usize; 1]>> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    
    Array::from(idx)
}


pub fn reorder_array<T: Copy>(array: &Array<T, Dim<[usize; 1]>>, order: &Array<usize, Dim<[usize; 1]>>) -> Array<T, Dim<[usize; 1]>> {
    let mut new_vec: Vec<T> = Vec::new();
    for i in order.iter() {
        new_vec.push(array[*i]);
    }

    Array::from(new_vec)
}


pub fn reorder_vec<T: Clone, U: IntoIterator<Item=usize>>(vec: &Vec<T>, order: U) -> Vec<T> {
    let new_vec: Vec<T> = order
        .into_iter()
        .map(|i| vec[i].clone())
        .collect();

    new_vec
}


pub fn array_from_vec_of_arrays<T: Clone>(vec: Vec<Array<T, Dim<[usize; 1]>>>) -> Array<T, Dim<[usize; 2]>> {
    let num_rows = vec.len();
    let num_cols = vec[0].len();
    let mut new_vec: Vec<T> = Vec::new();
    for arr in vec.into_iter() {
        for val in arr.into_iter() {
            new_vec.push(val.clone());
        }
    }

    Array::from_shape_vec((num_rows, num_cols), new_vec).unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_random_gaussian_vector() {
        let rand_vec = random_gaussian_vector(10, 0., 1.);
        assert_eq!(rand_vec.len(), 10);
    }

    #[test]
    fn test_random_gaussian_matrix() {
        let rand_matrix = random_gaussian_matrix(
            (4, 4), 
            0., 
            1.);

        assert_eq!(rand_matrix.dim(), (4 as usize, 4 as usize));
    }

    #[test]
    fn test_reorder_array() {
        let array = Array::from(vec![1,2,3]);
        let order = Array::from(vec![2, 1, 0]);
        let answer = Array::from(vec![3,2,1]);

        assert_eq!(reorder_array(&array, &order), answer);
    }

    #[test]
    fn test_reorder_vec() {
        let vec = vec![1, 2, 3];
        let order: Vec<usize> = vec![2, 1, 0];
        let answer = vec![3, 2, 1];

        assert_eq!(reorder_vec(&vec, order), answer);
    }
}