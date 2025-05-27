// This file contains GPU kernel implementations for various operations.
// The kernels are designed to be used with low-level GPU programming in Rust.

use std::os::raw::c_int;

#[no_mangle]
pub extern "C" fn relu_kernel(input: *const f32, output: *mut f32, length: c_int) {
    let input_slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length as usize) };

    for i in 0..length {
        output_slice[i as usize] = input_slice[i as usize].max(0.0);
    }
}

#[no_mangle]
pub extern "C" fn leaky_relu_kernel(input: *const f32, output: *mut f32, length: c_int, alpha: f32) {
    let input_slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length as usize) };

    for i in 0..length {
        output_slice[i as usize] = if input_slice[i as usize] > 0.0 {
            input_slice[i as usize]
        } else {
            alpha * input_slice[i as usize]
        };
    }
}

#[no_mangle]
pub extern "C" fn sigmoid_kernel(input: *const f32, output: *mut f32, length: c_int) {
    let input_slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length as usize) };

    for i in 0..length {
        output_slice[i as usize] = 1.0 / (1.0 + (-input_slice[i as usize]).exp());
    }
}

#[no_mangle]
pub extern "C" fn binary_cross_entropy_kernel(
    y_true: *const f32,
    y_pred: *const f32,
    length: c_int,
    output: *mut f32,
) {
    let y_true_slice = unsafe { std::slice::from_raw_parts(y_true, length as usize) };
    let y_pred_slice = unsafe { std::slice::from_raw_parts(y_pred, length as usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length as usize) };

    let epsilon = 1e-9;

    for i in 0..length {
        let pred = y_pred_slice[i as usize].clamp(epsilon, 1.0 - epsilon);
        output_slice[i as usize] = -y_true_slice[i as usize] * pred.ln() - (1.0 - y_true_slice[i as usize]) * (1.0 - pred).ln();
    }
}

#[no_mangle]
pub extern "C" fn kwta_kernel(input: *const f32, output: *mut f32, length: c_int, k: c_int) {
    let input_slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length as usize) };

    let mut indices: Vec<usize> = (0..length as usize).collect();
    indices.sort_by(|&a, &b| input_slice[b].partial_cmp(&input_slice[a]).unwrap());

    for i in 0..length {
        output_slice[i as usize] = if i < k {
            input_slice[indices[i as usize]]
        } else {
            0.0
        };
    }
}