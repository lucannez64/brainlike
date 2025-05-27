// This file implements the k-Winner-Take-All (kWTA) operation for GPU execution.
// It defines a function that selects the top k activations from an input array.

use std::cmp;
use std::slice;

pub fn kwta(input: &[f32], k: usize) -> Vec<f32> {
    let n = input.len();
    if k == 0 || k > n {
        return vec![0.0; n]; // Return a vector of zeros if k is invalid
    }

    // Create a vector of indices and sort them based on the input values
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| input[b].partial_cmp(&input[a]).unwrap());

    // Create a result vector initialized to zero
    let mut result = vec![0.0; n];

    // Set the top k activations in the result vector
    for i in 0..k {
        result[indices[i]] = input[indices[i]];
    }

    result
}