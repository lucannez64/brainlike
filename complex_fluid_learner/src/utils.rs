use anyhow::Result;
use ndarray::Array2;
use rand::prelude::*;

pub fn generate_synthetic_fluid_data(
    n_samples: usize,
    seed: Option<u64>,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let mut inputs = Array2::zeros((n_samples, 2));
    let mut outputs = Array2::zeros((n_samples, 2));

    for i in 0..n_samples {
        let param1 = rng.gen_range(0.0..1.0);
        let param2 = rng.gen_range(0.0..1.0) - 0.5;

        inputs[[i, 0]] = param1;
        inputs[[i, 1]] = param2;

        let noise1 = rng.gen_range(0.0..0.04) - 0.02;
        let noise2 = rng.gen_range(0.0..0.04) - 0.02;

        let output1 = 0.5 * (3.0 * std::f32::consts::PI * param1).sin()
            + param2.powi(3)
            + 0.2 * param1 * param2
            + noise1;

        let output2 = 0.3 * param1.powi(2)
            + param2.abs()
            + 0.1 * (2.0 * std::f32::consts::PI * param1).cos()
            + noise2;

        outputs[[i, 0]] = output1;
        outputs[[i, 1]] = output2;
    }

    // Normalize outputs to [0, 1]
    for j in 0..2 {
        let column = outputs.column(j);
        let min_val = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val + 1e-6;

        for i in 0..n_samples {
            outputs[[i, j]] = (outputs[[i, j]] - min_val) / range;
        }
    }

    Ok((inputs, outputs))
}

pub fn mean_squared_error(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> f32 {
    (y_true - y_pred).mapv(|x| x * x).mean().unwrap()
}

pub fn mean_absolute_error(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> f32 {
    (y_true - y_pred).mapv(|x| x.abs()).mean().unwrap()
}

pub fn r_squared(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> f32 {
    let ss_res = (y_true - y_pred).mapv(|x| x * x).sum();

    let y_mean = y_true.mean().unwrap();
    let ss_tot = y_true.mapv(|x| (x - y_mean).powi(2)).sum();

    if ss_tot == 0.0 {
        if ss_res == 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - (ss_res / ss_tot)
    }
}
