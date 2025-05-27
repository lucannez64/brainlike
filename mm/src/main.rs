// This is the entry point of the Rust project implementing the provided Python code.
// It initializes the GPU context and runs the main learning loop.

mod gpu;
mod ops;

use ndarray::{Axis, concatenate};
use ndarray::s;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize parameters for the learning process
    let num_reg_tasks = 2;
    let num_clf_tasks = 1;
    let learning_rate = 0.0005;
    let n_epochs = 100;

    // Load biological data directly from pickle files
    let mut inputs_list = Vec::new();
    let mut regs_list = Vec::new();
    let mut clfs_list = Vec::new();
    // Also load any CSV files in the `data/` directory
    for entry in std::fs::read_dir("data")? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("csv") {
            let (x, y_reg, y_clf) = ops::load_bio_data(
                path.to_str().unwrap(),
                num_reg_tasks,
                num_clf_tasks,
            )?;
            inputs_list.push(x);
            regs_list.push(y_reg);
            clfs_list.push(y_clf);
        }
    }
    let inputs = concatenate(Axis(0), &inputs_list.iter().map(|a| a.view()).collect::<Vec<_>>())?;
    let targets_reg = concatenate(Axis(0), &regs_list.iter().map(|a| a.view()).collect::<Vec<_>>())?;
    let targets_clf = concatenate(Axis(0), &clfs_list.iter().map(|a| a.view()).collect::<Vec<_>>())?;

    // Subsample to a maximum number of samples to fit in memory
    let max_samples = 1000000;
    let total_samples = inputs.shape()[0];
    let use_samples = total_samples.min(max_samples);
    let inputs = inputs.slice(s![0..use_samples, ..]).to_owned();
    let targets_reg = targets_reg.slice(s![0..use_samples, ..]).to_owned();
    let targets_clf = targets_clf.slice(s![0..use_samples, ..]).to_owned();

    // Split into training and validation sets (80/20)
    let total = inputs.shape()[0];
    let n_train = (total as f32 * 0.8) as usize;
    let train_inputs = inputs.slice(s![0..n_train, ..]).to_owned();
    let val_inputs   = inputs.slice(s![n_train.., ..]).to_owned();
    let train_targets_reg = targets_reg.slice(s![0..n_train, ..]).to_owned();
    let val_targets_reg   = targets_reg.slice(s![n_train.., ..]).to_owned();
    let train_targets_clf = targets_clf.slice(s![0..n_train, ..]).to_owned();
    let val_targets_clf   = targets_clf.slice(s![n_train.., ..]).to_owned();

    // Initialize deep 7-layer learner
    let hidden_sizes = [64, 64, 32, 32, 16, 16, 8];
    let mut deep = ops::DeepLearner7::new(
        inputs.shape()[1],
        &hidden_sizes,
        num_reg_tasks,
        num_clf_tasks,
        learning_rate,
    );

    // Train on biological data
    deep.learn(
        &inputs,
        &targets_reg,
        &targets_clf,
        n_epochs,
    );
    // Save weights
    deep.save_weights("deep7_weights.json")?;

    // Evaluate on full dataset
    let (pred_reg, pred_clf) = deep.predict(&inputs);
    ops::print_results(pred_reg, pred_clf);

    Ok(())
}