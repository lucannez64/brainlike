// This file contains implementations of loss functions used in the neural network operations.

use ndarray::Array1;

pub fn binary_cross_entropy(y_true: &Array1<f32>, y_pred: &Array1<f32>) -> f32 {
    let epsilon: f32 = 1e-7;
    let _ones = Array1::from_elem(y_true.len(), 1.0);
    let y_pred_clipped = y_pred.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
    let loss = -y_true.dot(&y_pred_clipped.mapv(|x| x.ln()))
        - (&_ones - y_true).dot(&(&_ones - &y_pred_clipped).mapv(|x| x.ln()));
    loss / (y_true.len() as f32)
}

pub fn mean_squared_error(y_true: &Array1<f32>, y_pred: &Array1<f32>) -> f32 {
    let diff = y_true - y_pred;
    diff.mapv(|x| x.powi(2)).mean().unwrap_or(0.0)
}

pub fn categorical_cross_entropy(y_true: &Array1<f32>, y_pred: &Array1<f32>) -> f32 {
    let epsilon: f32 = 1e-7;
    let _ones = Array1::from_elem(y_true.len(), 1.0);
    let y_pred_clipped = y_pred.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
    let loss = -y_true.dot(&y_pred_clipped.mapv(|x| x.ln()));
    loss / (y_true.len() as f32)
}