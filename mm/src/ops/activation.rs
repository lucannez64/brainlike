// This file contains implementations of activation functions for neural networks.
// The functions are designed to be used with low-level GPU operations.

use glam::Vec4;

pub fn relu(input: &mut [f32]) {
    let len = input.len();
    let rem = len % 4;
    for chunk in input[..len - rem].chunks_exact_mut(4) {
        let vec = Vec4::new(chunk[0], chunk[1], chunk[2], chunk[3]);
        let result = vec.max(Vec4::ZERO);
        let arr = result.to_array();
        chunk.copy_from_slice(&arr);
    }
    for v in &mut input[len - rem..] {
        *v = v.max(0.0);
    }
}

pub fn leaky_relu(input: &mut [f32], alpha: f32) {
    let len = input.len();
    let rem = len % 4;
    for chunk in input[..len - rem].chunks_exact_mut(4) {
        let vec = Vec4::new(chunk[0], chunk[1], chunk[2], chunk[3]);
        let arr = vec.to_array();
        let result = Vec4::new(
            if arr[0] > 0.0 { arr[0] } else { alpha * arr[0] },
            if arr[1] > 0.0 { arr[1] } else { alpha * arr[1] },
            if arr[2] > 0.0 { arr[2] } else { alpha * arr[2] },
            if arr[3] > 0.0 { arr[3] } else { alpha * arr[3] },
        );
        let out = result.to_array();
        chunk.copy_from_slice(&out);
    }
    for v in &mut input[len - rem..] {
        *v = if *v > 0.0 { *v } else { alpha * *v };
    }
}

pub fn sigmoid(input: &mut [f32]) {
    let len = input.len();
    let rem = len % 4;
    for chunk in input[..len - rem].chunks_exact_mut(4) {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = 1.0 / ((-chunk[i]).exp() + 1.0);
        }
        chunk.copy_from_slice(&out);
    }
    for v in &mut input[len - rem..] {
        *v = 1.0 / ((-*v).exp() + 1.0);
    }
}

pub fn binary_cross_entropy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    let mut total_loss = 0.0;
    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        let clipped_pred = pred_val.clamp(1e-9, 1.0 - 1e-9);
        total_loss -= true_val * clipped_pred.ln() + (1.0 - true_val) * (1.0 - clipped_pred).ln();
    }
    total_loss / y_true.len() as f32
}