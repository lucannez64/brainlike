mod gpu_context;
mod micro_circuit;
mod neural_network;
mod shaders;
mod utils;

use anyhow::Result;
use ndarray::s;
use neural_network::ComplexFluidLearner;
use utils::{generate_synthetic_fluid_data, r_squared};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Complex Fluid Learner - Rust GPU Implementation");
    println!("{}", "=".repeat(70));

    // Configuration - reduced epochs for demo
    const NUM_SAMPLES: usize = 10000;
    const TRAIN_SPLIT_RATIO: f32 = 0.8;
    const VALIDATION_SPLIT_TRAIN: f32 = 0.15;
    const N_HIDDEN_CIRCUITS: usize = 24;
    const N_INTERNAL_UNITS: usize = 7;
    const LEARNING_RATE: f32 = 0.002;
    const N_EPOCHS: usize = 100; // Reduced from 50000 for demo
    const BATCH_SIZE: usize = 64;
    const MIN_EPOCHS_NO_IMPROVE: usize = 15; // Reduced from 150
    const PATIENCE_NO_IMPROVE: usize = 30; // Reduced from 300

    // Generate synthetic data
    let (input_features, target_outputs) = generate_synthetic_fluid_data(NUM_SAMPLES, Some(123))?;

    let n_train_val = (NUM_SAMPLES as f32 * TRAIN_SPLIT_RATIO) as usize;
    let train_val_x = input_features.slice(s![..n_train_val, ..]).to_owned();
    let train_val_y = target_outputs.slice(s![..n_train_val, ..]).to_owned();
    let test_x = input_features.slice(s![n_train_val.., ..]).to_owned();
    let test_y = target_outputs.slice(s![n_train_val.., ..]).to_owned();

    println!("📊 Data Overview:");
    println!("   Total samples: {}", NUM_SAMPLES);
    println!("   Training/Validation: {:?}", train_val_x.shape());
    println!("   Testing: {:?}", test_x.shape());

    let n_inputs = train_val_x.shape()[1];
    let n_outputs = train_val_y.shape()[1];

    // Initialize learner
    println!("\n🏗️  Initializing Neural Network...");
    let mut learner = ComplexFluidLearner::new(
        n_inputs,
        n_outputs,
        N_HIDDEN_CIRCUITS,
        N_INTERNAL_UNITS,
        LEARNING_RATE,
        Some(42),
    )
    .await?;

    println!(
        "   Network Architecture: {} → {} → {}",
        n_inputs, N_HIDDEN_CIRCUITS, n_outputs
    );
    println!(
        "   Micro-circuits: {} circuits × {} units each",
        N_HIDDEN_CIRCUITS, N_INTERNAL_UNITS
    );
    println!(
        "   Learning rate: {}, Max epochs: {}",
        LEARNING_RATE, N_EPOCHS
    );

    // Training
    println!("\n🚀 Starting Training...");
    learner
        .train(
            &train_val_x,
            &train_val_y,
            N_EPOCHS,
            BATCH_SIZE,
            VALIDATION_SPLIT_TRAIN,
            MIN_EPOCHS_NO_IMPROVE,
            PATIENCE_NO_IMPROVE,
        )
        .await?;

    // Evaluation
    println!("\n🧪 Evaluating on Test Data...");
    if !test_x.is_empty() {
        let test_predictions = learner.predict(&test_x, BATCH_SIZE).await?;
        let mse_test = utils::mean_squared_error(&test_y, &test_predictions);
        let mae_test = utils::mean_absolute_error(&test_y, &test_predictions);
        let r2_test = r_squared(&test_y, &test_predictions);

        println!("\n📈 Test Results:");
        println!("   Mean Squared Error (MSE): {:.6}", mse_test);
        println!("   Mean Absolute Error (MAE): {:.6}", mae_test);
        println!("   R-squared (overall): {:.6}", r2_test);

        for i in 0..n_outputs {
            let y_true = test_y.column(i);
            let y_pred = test_predictions.column(i);
            let r2_output = r_squared(
                &y_true.to_owned().insert_axis(ndarray::Axis(1)),
                &y_pred.to_owned().insert_axis(ndarray::Axis(1)),
            );
            println!("   R-squared (Output {}): {:.6}", i + 1, r2_output);
        }
    }

    println!("\n✅ Demo Completed Successfully!");
    Ok(())
}
