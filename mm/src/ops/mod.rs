pub mod activation;
pub mod kwta;
pub mod loss;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};
use std::ops::AddAssign;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::s;
use plotters::prelude::*;
use serde::Serialize;
use std::fs::File;
use csv::Reader;
use ndarray::{Array3};
use serde_pickle::value::{HashableValue, Value};
use std::collections::BTreeMap;
use ndarray_rand::rand::thread_rng;

/// Generate synthetic data stub
pub fn generate_data(num_samples: usize, num_reg_tasks: usize, num_clf_tasks: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let mut rng = thread_rng();
    // Generate base features
    let x0: Array1<f32> = Array1::random_using(num_samples, Uniform::new(-2.0, 2.0), &mut rng);
    let x1: Array1<f32> = Array1::random_using(num_samples, Uniform::new(-2.0, 2.0), &mut rng);
    let x2: Array1<f32> = Array1::random_using(num_samples, Uniform::new(0.0, 1.0), &mut rng);
    // Nonlinear combinations
    let x3: Array1<f32> = x0.iter().zip(x1.iter())
        .map(|(&a, &b)| (std::f32::consts::PI * a * 0.8).sin() + 0.5 * (std::f32::consts::PI * b * 0.6).cos())
        .collect::<Array1<f32>>();
    let x4: Array1<f32> = x0.iter().zip(x1.iter()).zip(x2.iter())
        .map(|((&a, &b), &c)| a * b * (-0.1 * c * c).exp())
        .collect::<Array1<f32>>();
    let x5: Array1<f32> = Array1::random_using(num_samples, Normal::new(0.0, 0.3).unwrap(), &mut rng);
    // Stack into (num_samples, 6)
    let inputs = ndarray::stack![Axis(1), x0.view(), x1.view(), x2.view(), x3.view(), x4.view(), x5.view()];
    // Regression targets
    let mut targets_reg = Array2::zeros((num_samples, num_reg_tasks));
    if num_reg_tasks > 0 { targets_reg.column_mut(0).assign(&x0); }
    if num_reg_tasks > 1 { targets_reg.column_mut(1).assign(&x1); }
    if num_reg_tasks > 2 { targets_reg.column_mut(2).assign(&x2); }
    // Classification targets (binary from x0 > 0)
    let mut targets_clf = Array2::zeros((num_samples, num_clf_tasks));
    for j in 0..num_clf_tasks {
        for (i, &val0) in x0.iter().enumerate() {
            // Create diverse classification tasks
            let label = match j {
                0 => if val0 > 0.0 { 1.0 } else { 0.0 },
                1 => if x1[i] > 0.5 { 1.0 } else { 0.0 },
                2 => if x2[i] < 0.3 { 1.0 } else { 0.0 },
                _ => if val0 + x1[i] - x2[i] > 0.0 { 1.0 } else { 0.0 },
            };
            targets_clf[(i, j)] = label;
        }
    }
    (inputs, targets_reg, targets_clf)
}

/// Multi-task learner implementation inspired by Python version
pub struct MultiTaskComplexLearner {
    n_inputs: usize,
    num_reg_tasks: usize,
    num_clf_tasks: usize,
    lr: f32,
    // Layer 1 weights and bias: (n_inputs -> hidden_size)
    w1: Array2<f32>,
    b1: Array1<f32>,
    // Output layer weights and biases
    w2_reg: Array2<f32>,
    b2_reg: Array1<f32>,
    w2_clf: Array2<f32>,
    b2_clf: Array1<f32>,
    // History
    pub train_loss: Vec<f32>,
    pub train_loss_reg: Vec<f32>,
    pub train_loss_clf: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub val_loss_reg: Vec<f32>,
    pub val_loss_clf: Vec<f32>,
}

impl MultiTaskComplexLearner {
    pub fn new(n_inputs: usize, num_reg_tasks: usize, num_clf_tasks: usize, learning_rate: f32) -> Self {
        let hidden_size = 40; // choose hidden dimension
        let mut rng = thread_rng();
        let w1 = Array2::random_using((n_inputs, hidden_size), Uniform::new(-0.5, 0.5), &mut rng);
        let b1 = Array1::zeros(hidden_size);
        let w2_reg = Array2::random_using((hidden_size, num_reg_tasks), Uniform::new(-0.5, 0.5), &mut rng);
        let b2_reg = Array1::zeros(num_reg_tasks);
        let w2_clf = Array2::random_using((hidden_size, num_clf_tasks), Uniform::new(-0.5, 0.5), &mut rng);
        let b2_clf = Array1::zeros(num_clf_tasks);
        MultiTaskComplexLearner { n_inputs, num_reg_tasks, num_clf_tasks, lr: learning_rate, w1, b1, w2_reg, b2_reg, w2_clf, b2_clf, train_loss: Vec::new(), train_loss_reg: Vec::new(), train_loss_clf: Vec::new(), val_loss: Vec::new(), val_loss_reg: Vec::new(), val_loss_clf: Vec::new() }
    }

    /// Forward pass: returns hidden activations, regression & classification outputs
    fn forward(&self, inputs: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // Hidden pre-activation
        let mut hidden = inputs.dot(&self.w1);
        hidden.add_assign(&self.b1);
        // ReLU activation
        hidden.mapv_inplace(|x| x.max(0.0));
        // Regression output
        let mut out_reg = hidden.dot(&self.w2_reg);
        out_reg.add_assign(&self.b2_reg);
        // Classification logits -> sigmoid
        let logits = hidden.dot(&self.w2_clf);
        let mut out_clf = logits;
        out_clf.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        (hidden, out_reg, out_clf)
    }

    /// Compute MSE + BCE loss
    fn compute_loss(&self, y_true_reg: &Array2<f32>, y_pred_reg: &Array2<f32>, y_true_clf: &Array2<f32>, y_pred_clf: &Array2<f32>) -> f32 {
        let mse = loss::mean_squared_error(&y_true_reg.column(0).to_owned(), &y_pred_reg.column(0).to_owned());
        let mut bce = 0.0;
        if self.num_clf_tasks > 0 {
            bce = loss::binary_cross_entropy(&y_true_clf.column(0).to_owned(), &y_pred_clf.column(0).to_owned());
        }
        mse + bce
    }

    /// Train the model for n_epochs
    pub fn learn(&mut self, inputs: &Array2<f32>, targets_reg: &Array2<f32>, targets_clf: &Array2<f32>, n_epochs: usize) {
        let n_total = inputs.shape()[0];
        let val_size = (n_total as f32 * 0.1) as usize;
        let train_size = n_total - val_size;
        let train_X = inputs.slice(s![..train_size, ..]).to_owned();
        let train_Y_reg = targets_reg.slice(s![..train_size, ..]).to_owned();
        let train_Y_clf = targets_clf.slice(s![..train_size, ..]).to_owned();
        let val_X = inputs.slice(s![train_size.., ..]).to_owned();
        let val_Y_reg = targets_reg.slice(s![train_size.., ..]).to_owned();
        let val_Y_clf = targets_clf.slice(s![train_size.., ..]).to_owned();
        let n = train_size as f32;
        // Progress bar setup
        let pb = ProgressBar::new(n_epochs as u64);
        pb.set_style(
            ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} Epoch {msg}")
                .unwrap()
                .progress_chars("█▇▆▅▃▂▁  ")
        );
        for _epoch in 1..=n_epochs {
            // Forward pass
            let (hidden, pred_reg, pred_clf) = self.forward(&train_X);
            // Compute loss
            // Train metrics
            let loss_reg_t = loss::mean_squared_error(&train_Y_reg.column(0).to_owned(), &pred_reg.column(0).to_owned());
            let loss_clf_t = if self.num_clf_tasks>0 { loss::binary_cross_entropy(&train_Y_clf.column(0).to_owned(), &pred_clf.column(0).to_owned()) } else {0.0};
            let loss_t = loss_reg_t + loss_clf_t;
            // Val metrics
            let (_, pred_reg_val, pred_clf_val) = self.forward(&val_X);
            let loss_reg_v = loss::mean_squared_error(&val_Y_reg.column(0).to_owned(), &pred_reg_val.column(0).to_owned());
            let loss_clf_v = if self.num_clf_tasks>0 { loss::binary_cross_entropy(&val_Y_clf.column(0).to_owned(), &pred_clf_val.column(0).to_owned()) } else {0.0};
            let loss_v = loss_reg_v + loss_clf_v;
            // Record history
            self.train_loss.push(loss_t);
            self.train_loss_reg.push(loss_reg_t);
            self.train_loss_clf.push(loss_clf_t);
            self.val_loss.push(loss_v);
            self.val_loss_reg.push(loss_reg_v);
            self.val_loss_clf.push(loss_clf_v);
            // Update progress bar message and advance
            pb.set_message(format!("train={:.4}, val={:.4}", loss_t, loss_v));
            pb.inc(1);
            // Backward pass
            // Regression gradient: dL/dy = 2*(pred - true)/n
            let d_reg = (&pred_reg - &train_Y_reg) * (2.0 / n);
            // Classification gradient: dL/dlogits = (pred - true)/n
            let d_clf = (&pred_clf - &train_Y_clf) * (1.0 / n);
            // Gradients w.r.t. output weights
            let grad_w2_reg = hidden.t().dot(&d_reg);
            let grad_b2_reg = d_reg.sum_axis(Axis(0));
            let grad_w2_clf = hidden.t().dot(&d_clf);
            let grad_b2_clf = d_clf.sum_axis(Axis(0));
            // Backprop into hidden layer
            let mut hidden_err = d_reg.dot(&self.w2_reg.t());
            hidden_err += &d_clf.dot(&self.w2_clf.t());
            // ReLU derivative
            let relu_deriv = hidden.mapv(|x| if x > 0.0 {1.0} else {0.0});
            let hidden_grad = &hidden_err * &relu_deriv;
            // Gradients w.r.t. first layer (using training subset)
            let grad_w1 = train_X.t().dot(&hidden_grad);
            let grad_b1 = hidden_grad.sum_axis(Axis(0));
            // Update parameters (SGD)
            self.w1 -= &(grad_w1 * self.lr);
            self.b1 -= &(grad_b1 * self.lr);
            self.w2_reg -= &(grad_w2_reg * self.lr);
            self.b2_reg -= &(grad_b2_reg * self.lr);
            self.w2_clf -= &(grad_w2_clf * self.lr);
            self.b2_clf -= &(grad_b2_clf * self.lr);
        }
        pb.finish_with_message("Training complete");
    }

    /// Plot training and validation losses over epochs
    pub fn plot_history(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let epochs: Vec<i32> = (1..=self.train_loss.len()).map(|x| x as i32).collect();
        let mut chart = ChartBuilder::on(&root)
            .caption("Training vs Validation Loss", ("sans-serif", 20).into_font().color(&BLACK))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(1i32..(*epochs.last().unwrap()), 0f32..self.train_loss.iter().chain(self.val_loss.iter()).cloned().fold(f32::NEG_INFINITY, f32::max))?;
        chart.configure_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            epochs.iter().zip(self.train_loss.iter()).map(|(&x,y)| (x, *y)),
            ShapeStyle::from(&RED).stroke_width(2),
        ))?.label("Train").legend(|(x,y)| PathElement::new(vec![(x,y),(x+20,y)], &RED));
        chart.draw_series(LineSeries::new(
            epochs.iter().zip(self.val_loss.iter()).map(|(&x,y)| (x, *y)),
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))?.label("Val").legend(|(x,y)| PathElement::new(vec![(x,y),(x+20,y)], &BLUE));
        chart.configure_series_labels().border_style(&BLACK).draw()?;
        Ok(())
    }

    /// Plot separate regression and classification loss history
    pub fn plot_component_losses(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(filename, (800, 400)).into_drawing_area();
        root.fill(&WHITE)?;
        let epochs: Vec<i32> = (1..=self.train_loss_reg.len()).map(|x| x as i32).collect();
        let max_y = self.train_loss_reg.iter().chain(self.train_loss_clf.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut chart = ChartBuilder::on(&root)
            .caption("Regression vs Classification Train Loss", ("sans-serif", 16).into_font().color(&BLACK))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(1i32..(*epochs.last().unwrap()), 0f32..max_y)?;
        chart.configure_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            epochs.iter().zip(self.train_loss_reg.iter()).map(|(&x,y)| (x, *y)),
            ShapeStyle::from(&GREEN).stroke_width(2),
        ))?.label("Train Reg").legend(|(x,y)| PathElement::new(vec![(x,y),(x+20,y)], &GREEN));
        chart.draw_series(LineSeries::new(
            epochs.iter().zip(self.train_loss_clf.iter()).map(|(&x,y)| (x, *y)),
            ShapeStyle::from(&MAGENTA).stroke_width(2),
        ))?.label("Train Clf").legend(|(x,y)| PathElement::new(vec![(x,y),(x+20,y)], &MAGENTA));
        chart.configure_series_labels().border_style(&BLACK).draw()?;
        Ok(())
    }

    /// Scatter plot of true vs predicted regression values for the first task
    pub fn plot_regression_scatter(&self, true_y: &Array2<f32>, pred_y: &Array2<f32>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(filename, (400, 400)).into_drawing_area();
        root.fill(&WHITE)?;
        let data_true = true_y.column(0);
        let data_pred = pred_y.column(0);
        let min_val = data_true.iter().chain(data_pred.iter()).cloned().fold(f32::INFINITY, f32::min);
        let max_val = data_true.iter().chain(data_pred.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut chart = ChartBuilder::on(&root)
            .caption("True vs Predicted Regression", ("sans-serif", 16).into_font().color(&BLACK))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_val..max_val, min_val..max_val)?;
        chart.configure_mesh().draw()?;
        chart.draw_series(data_true.iter().zip(data_pred.iter()).map(|(&x,&y)| Circle::new((x, y), 2, ShapeStyle::from(&BLUE).filled())))?;
        Ok(())
    }

    /// Histogram of predicted classification probabilities for the first task
    pub fn plot_classification_hist(&self, pred_clf: &Array2<f32>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(filename, (400, 300)).into_drawing_area();
        root.fill(&WHITE)?;
        let data: Vec<f32> = pred_clf.column(0).to_vec();
        let bins = 20;
        let min_val = 0.0;
        let max_val = 1.0;
        let bin_width = (max_val - min_val) / bins as f32;
        let mut counts = vec![0; bins];
        for &v in &data { let idx = ((v - min_val) / bin_width).floor().clamp(0.0, (bins-1) as f32) as usize; counts[idx]+=1; }
        let mut chart = ChartBuilder::on(&root)
            .caption("Histogram of Predicted Probabilities", ("sans-serif", 16).into_font().color(&BLACK))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_val..max_val, 0..*counts.iter().max().unwrap())?;
        chart.configure_mesh().draw()?;
        chart.draw_series(counts.iter().enumerate().map(|(i,&c)| Rectangle::new([
            (min_val + i as f32*bin_width, 0),
            (min_val + (i+1) as f32*bin_width, c),
        ], ShapeStyle::from(&RED).filled())))?;
        Ok(())
    }

    /// Predict using learned weights
    pub fn predict(&self, inputs: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (_hidden, out_reg, out_clf) = self.forward(inputs);
        (out_reg, out_clf)
    }

    /// Save model weights and biases to JSON
    pub fn save_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Serialize)] struct Weights { w1: Vec<Vec<f32>>, b1: Vec<f32>, w2_reg: Vec<Vec<f32>>, b2_reg: Vec<f32>, w2_clf: Vec<Vec<f32>>, b2_clf: Vec<f32> }
        let w1 = self.w1.rows().into_iter().map(|r| r.to_vec()).collect();
        let b1 = self.b1.to_vec();
        let w2_reg = self.w2_reg.rows().into_iter().map(|r| r.to_vec()).collect();
        let b2_reg = self.b2_reg.to_vec();
        let w2_clf = self.w2_clf.rows().into_iter().map(|r| r.to_vec()).collect();
        let b2_clf = self.b2_clf.to_vec();
        let ws = Weights { w1, b1, w2_reg, b2_reg, w2_clf, b2_clf };
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &ws)?;
        Ok(())
    }
}

/// 7-Layer Deep Multi-Task Learner
pub struct DeepLearner7 {
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
    w_out_reg: Array2<f32>,
    b_out_reg: Array1<f32>,
    w_out_clf: Array2<f32>,
    b_out_clf: Array1<f32>,
    lr: f32,
}

impl DeepLearner7 {
    /// Build a 7-layer MLP with specified hidden sizes
    pub fn new(n_inputs: usize, hidden_sizes: &[usize;7], num_reg: usize, num_clf: usize, lr: f32) -> Self {
        let mut rng = thread_rng();
        let mut in_dim = n_inputs;
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for &h in hidden_sizes.iter() {
            // smaller initial weights for stability
            weights.push(Array2::random_using((in_dim, h), Uniform::new(-0.1,0.1), &mut rng));
         biases.push(Array1::zeros(h));
         in_dim = h;
         }
        let w_out_reg = Array2::random_using((in_dim, num_reg), Uniform::new(-0.1,0.1), &mut rng);
        let b_out_reg = Array1::zeros(num_reg);
        let w_out_clf = Array2::random_using((in_dim, num_clf), Uniform::new(-0.1,0.1), &mut rng);
        let b_out_clf = Array1::zeros(num_clf);
        DeepLearner7 { weights, biases, w_out_reg, b_out_reg, w_out_clf, b_out_clf, lr }
    }

    /// Forward pass for DeepLearner7, returning all hidden activations
    fn forward(&self, inputs: &Array2<f32>) -> (Vec<Array2<f32>>, Array2<f32>, Array2<f32>) {
        let mut activations = Vec::with_capacity(self.weights.len() + 1);
        // input as activation[0]
        activations.push(inputs.clone());
        // hidden layers
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let mut h = activations.last().unwrap().dot(w);
            h.add_assign(b);
            h.mapv_inplace(|u| u.max(0.0));
            activations.push(h);
        }
        // last hidden activation
        let last = activations.last().unwrap();
        // regression output
        let mut out_reg = last.dot(&self.w_out_reg);
        out_reg.add_assign(&self.b_out_reg);
        // classification output
        let mut logits_clf = last.dot(&self.w_out_clf);
        logits_clf.add_assign(&self.b_out_clf);
        let out_clf = logits_clf.mapv(|u| 1.0 / (1.0 + (-u).exp()));
        (activations, out_reg, out_clf)
    }

    /// Train the model (full 7-layer MLP training loop)
    pub fn learn(&mut self,
         inputs: &Array2<f32>, targets_reg: &Array2<f32>, targets_clf: &Array2<f32>,
         epochs: usize) {
        let n = inputs.shape()[0] as f32;
        let pb = ProgressBar::new(epochs as u64);
        pb.set_style(
            ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} Epoch {msg}")
                .unwrap()
                .progress_chars("█▇▆▅▃▂▁  ")
        );
        for epoch in 1..=epochs {
            // forward with all activations
            let (acts, logits_reg, logits_clf) = self.forward(inputs);
            let last_act = &acts[acts.len()-1];
            let out_clf = logits_clf.mapv(|u| 1.0 / (1.0 + (-u).exp()));
            // compute losses
            let mse = (&logits_reg - targets_reg).mapv(|d| d*d).sum() / n;
            let bce = targets_clf * &out_clf.mapv(|p| p.max(1e-6).ln())
                      + (1.0 - targets_clf) * &out_clf.mapv(|p| (1.0 - p).max(1e-6).ln());
            let loss = mse - bce.sum() / n;
            pb.set_message(format!("Epoch {}: Loss={:.4}", epoch, loss));
            // output layer gradients
            let d_reg = (&logits_reg - targets_reg) * (2.0 / n);
            let d_clf = (&out_clf - targets_clf) * (1.0 / n);
            let grad_w_out_reg = last_act.t().dot(&d_reg);
            let grad_b_out_reg = d_reg.sum_axis(Axis(0));
            let grad_w_out_clf = last_act.t().dot(&d_clf);
            let grad_b_out_clf = d_clf.sum_axis(Axis(0));
            // update outputs
            self.w_out_reg -= &(grad_w_out_reg * self.lr);
            self.b_out_reg -= &(grad_b_out_reg * self.lr);
            self.w_out_clf -= &(grad_w_out_clf * self.lr);
            self.b_out_clf -= &(grad_b_out_clf * self.lr);
            // backpropagate through hidden layers
            let mut delta = d_reg.dot(&self.w_out_reg.t()) + d_clf.dot(&self.w_out_clf.t());
            for i in (0..self.weights.len()).rev() {
                // compute derivative of activation_i+1
                let deriv = acts[i+1].mapv(|u| if u > 0.0 {1.0} else {0.0});
                delta = delta * deriv;
                // gradient w.r.t. weights[i]
                let inp = &acts[i];
                let grad_w = inp.t().dot(&delta);
                let grad_b = delta.sum_axis(Axis(0));
                // update hidden weights and biases
                self.weights[i] -= &(grad_w * self.lr);
                self.biases[i] -= &(grad_b * self.lr);
                // propagate delta to previous layer
                delta = delta.dot(&self.weights[i].t());
            }
            pb.inc(1);
        }
        pb.finish_with_message("DeepLearner7 training complete");
    }

    /// Predict using learned weights
    pub fn predict(&self, inputs: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (_hidden, out_reg, out_clf) = self.forward(inputs);
        (out_reg, out_clf)
    }

    /// Serialize weights for DeepLearner7
    pub fn save_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Serialize)]
        struct Weights7 {
            hidden_ws: Vec<Vec<Vec<f32>>>,
            hidden_bs: Vec<Vec<f32>>,
            w_out_reg: Vec<Vec<f32>>,
            b_out_reg: Vec<f32>,
            w_out_clf: Vec<Vec<f32>>,
            b_out_clf: Vec<f32>,
        }
        let hidden_ws = self
            .weights
            .iter()
            .map(|m| m.rows().into_iter().map(|r| r.to_vec()).collect())
            .collect();
        let hidden_bs = self.biases.iter().map(|b| b.to_vec()).collect();
        let w_out_reg = self
            .w_out_reg
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        let b_out_reg = self.b_out_reg.to_vec();
        let w_out_clf = self
            .w_out_clf
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        let b_out_clf = self.b_out_clf.to_vec();
        let ws = Weights7 {
            hidden_ws,
            hidden_bs,
            w_out_reg,
            b_out_reg,
            w_out_clf,
            b_out_clf,
        };
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &ws)?;
        Ok(())
    }

    /// Load saved weights for DeepLearner7 from JSON file
    pub fn load_weights(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let ws: Weights7 = serde_json::from_reader(file)?;
        // restore hidden layer parameters
        for (i, w_mat) in ws.hidden_ws.iter().enumerate() {
            self.weights[i] = Array2::from_shape_vec((w_mat.len(), w_mat[0].len()), w_mat.concat())?;
            self.biases[i]  = Array1::from_vec(ws.hidden_bs[i].clone());
        }
        // restore outputs
        self.w_out_reg = Array2::from_shape_vec((ws.w_out_reg.len(), ws.w_out_reg[0].len()), ws.w_out_reg.concat())?;
        self.b_out_reg = Array1::from_vec(ws.b_out_reg);
        self.w_out_clf = Array2::from_shape_vec((ws.w_out_clf.len(), ws.w_out_clf[0].len()), ws.w_out_clf.concat())?;
        self.b_out_clf = Array1::from_vec(ws.b_out_clf);
        Ok(())
    }
}

/// A network of pretrained DeepLearner7 neurons connected in configurable layers
pub struct BioNetwork {
    layers: Vec<Vec<DeepLearner7>>,  
    synapses: Vec<Array2<f32>>, // weights between layers
}

impl BioNetwork {
    /// Build a new BioNetwork with given layer sizes. Each neuron is a DeepLearner7 initialized and optionally loaded from weights.
    pub fn new(
        layer_sizes: &[usize],
        hidden_sizes: &[usize;7],
        num_reg: usize,
        num_clf: usize,
        neuron_lr: f32,
        conn_lr: f32,
    ) -> Self {
        let mut layers = Vec::new();
        for &size in layer_sizes.iter() {
            let mut layer = Vec::new();
            for _ in 0..size {
                // initialize each neuron model
                let neuron = DeepLearner7::new(layer_sizes[0], hidden_sizes, num_reg, num_clf, neuron_lr);
                layer.push(neuron);
            }
            layers.push(layer);
        }
        // initialize synaptic weights between consecutive layers
        let mut synapses = Vec::new();
        for w in layer_sizes.windows(2) {
            let (in_dim, out_dim) = (w[0], w[1]);
            let mat = Array2::random_using((in_dim, out_dim), Uniform::new(-0.1, 0.1), &mut thread_rng());
            synapses.push(mat);
        }
        BioNetwork { layers, synapses }
    }

    /// Forward propagate an input vector through the network, returns final layer activations
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut activations = input.clone();
        for (layer, conn) in self.layers.iter().zip(self.synapses.iter()) {
            // weighted sum for next layer inputs
            let next_input = activations.dot(conn);
            // neuron outputs
            let mut next_act = Array1::<f32>::zeros(layer.len());
            for (i, neuron) in layer.iter().enumerate() {
                // DNN expects 2D array: single sample
                let sample = next_input.slice(s![i..=i]).to_owned().insert_axis(ndarray::Axis(0));
                let (out_reg, _) = neuron.predict(&sample);
                next_act[i] = out_reg[[0, 0]];
            }
            activations = next_act;
        }
        activations
    }
}

/// Print results stub
pub fn print_results(pred_reg: Array2<f32>, pred_clf: Array2<f32>) {
    println!("Regression predictions: {:?}", pred_reg);
    println!("Classification predictions: {:?}", pred_clf);
}

/// Load biological neuron data from CSV with columns: sim_id, time_index, features…, regression targets, classification labels
pub fn load_bio_data(path: &str, num_reg: usize, num_clf: usize)
    -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn std::error::Error>> {
    let mut rdr = Reader::from_path(path)?;
    let mut feat_rows = Vec::new();
    let mut reg_rows = Vec::new();
    let mut clf_rows = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let row: Vec<f32> = rec.iter().map(|s| s.parse().unwrap_or(0.0)).collect();
        // Compute number of feature inputs by total columns minus metadata and targets
        let total_cols = row.len();
        let n_inputs = total_cols - 2 - num_reg - num_clf;
        // Skip first two metadata columns and extract features
        let mut idx = 2;
        let feat = row[idx..idx + n_inputs].to_vec(); idx += n_inputs;
        // Extract regression targets
        let reg = row[idx..idx + num_reg].to_vec(); idx += num_reg;
        // Extract classification labels
        let clf = row[idx..idx + num_clf].to_vec();
        feat_rows.push(feat);
        reg_rows.push(reg);
        clf_rows.push(clf);
    }
    let n = feat_rows.len();
    let inputs = Array2::from_shape_vec((n, /*n_inputs*/ feat_rows[0].len()), feat_rows.concat())?;
    let targets_reg = Array2::from_shape_vec((n, num_reg), reg_rows.concat())?;
    let targets_clf = Array2::from_shape_vec((n, num_clf), clf_rows.concat())?;
    Ok((inputs, targets_reg, targets_clf))
}

/// Directly load a `.p` pickle file into feature, regression, and classification matrices
pub fn load_bio_pickle(path: &str) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn std::error::Error>> {
    // Read and parse pickle
    let file = File::open(path)?;
    let data: Value = serde_pickle::from_reader(file)?;
    // Expect top-level dict
    let top = match data {
        Value::Dict(map) => map,
        _ => return Err("Expected top-level dict".into()),
    };
    // helper to get entry by string key
    fn get<'a>(m: &'a BTreeMap<HashableValue, Value>, key: &str) -> Option<&'a Value> {
        m.get(&HashableValue::String(key.to_string()))
    }
    // Params
    let params = match get(&top, "Params") {
        Some(Value::Dict(pm)) => pm,
        _ => return Err("Missing Params".into()),
    };
    // Number of segments
    let num_seg = if let Some(Value::List(types)) = get(params, "allSegmentsType") {
        types.len()
    } else { return Err("Missing allSegmentsType".into()); };
    // Duration in ms
    let duration_ms = if let Some(Value::F64(sec)) = get(params, "totalSimDurationInSec") {
        (sec * 1000.0) as usize
    } else { return Err("Missing totalSimDurationInSec".into()); };
    // Results
    let results = match get(&top, "Results") {
        Some(Value::Dict(rm)) => rm,
        _ => return Err("Missing Results".into()),
    };
    // List of simulation dicts
    let sims = if let Some(Value::List(list)) = get(results, "listOfSingleSimulationDicts") {
        list
    } else { return Err("Missing listOfSingleSimulationDicts".into()); };
    let n_sims = sims.len();
    // allocate arrays
    let mut X3 = Array3::<f32>::zeros((n_sims, duration_ms, num_seg * 2));
    let mut Yreg = Array2::<f32>::zeros((n_sims, duration_ms));
    let mut Yclf = Array2::<f32>::zeros((n_sims, duration_ms));
    // iterate simulations
    for (i, sim_val) in sims.iter().enumerate() {
        let sim = match sim_val { Value::Dict(sm) => sm, _ => return Err("Invalid sim entry".into()) };
        // Ex spikes
        if let Some(Value::Dict(ex_map)) = get(sim, "exInputSpikeTimes") {
            for (k, v) in ex_map {
                if let (HashableValue::I64(seg), Value::List(times)) = (k, v) {
                    let s = *seg as usize;
                    for t_v in times {
                        if let Value::I64(t) = t_v {
                            let t = *t as usize;
                            if t < duration_ms { X3[[i, t, s]] = 1.0; }
                        }
                    }
                }
            }
        }
        // Inh spikes
        if let Some(Value::Dict(inh_map)) = get(sim, "inhInputSpikeTimes") {
            for (k, v) in inh_map {
                if let (HashableValue::I64(seg), Value::List(times)) = (k, v) {
                    let s = *seg as usize;
                    for t_v in times {
                        if let Value::I64(t) = t_v {
                            let t = *t as usize;
                            if t < duration_ms { X3[[i, t, num_seg + s]] = 1.0; }
                        }
                    }
                }
            }
        }
        // Soma voltage (regression targets)
        if let Some(Value::List(volts)) = get(sim, "somaVoltageLowRes") {
            for (t, vv) in volts.iter().enumerate() {
                if let Value::F64(v) = vv { Yreg[[i, t]] = *v as f32; }
            }
        }
        // Output spikes (classification)
        if let Some(Value::List(spks)) = get(sim, "outputSpikeTimes") {
            for vv in spks {
                if let Value::F64(v) = vv {
                    let t = ((v - 0.5).round() as usize).min(duration_ms - 1);
                    Yclf[[i, t]] = 1.0;
                }
            }
        }
    }
    // Flatten to 2D and return
    let X2 = X3.into_shape((n_sims * duration_ms, num_seg * 2))?;
    let reg2 = Yreg.into_shape((n_sims * duration_ms, 1))?;
    let clf2 = Yclf.into_shape((n_sims * duration_ms, 1))?;
    Ok((X2, reg2, clf2))
}