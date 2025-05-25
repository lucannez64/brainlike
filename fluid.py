import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict
from numba import jit

# --- Utility Functions (Decorate with @jit) ---
@jit(nopython=True, cache=True)
def sigmoid_numba(x: np.ndarray) -> np.ndarray:
    # Numba handles np.exp well. Clipping might be done before if extreme values are an issue.
    return 1.0 / (1.0 + np.exp(-x))

@jit(nopython=True, cache=True)
def sigmoid_derivative_numba(x_activated: np.ndarray) -> np.ndarray:
    return x_activated * (1.0 - x_activated)

# --- MicroCircuit Class (remains the same, its logic is now mostly inlined or called by jitted static methods) ---
class MicroCircuit:
    def __init__(self, n_internal_units: int = 3, input_scale: float = 1.0):
        self.n_internal_units = n_internal_units
        self.internal_weights = np.random.randn(n_internal_units).astype(np.float64) * input_scale
        self.internal_biases = np.random.randn(n_internal_units).astype(np.float64) * 0.5
        
    # These methods are not directly called by the jitted path anymore,
    # but kept for potential non-jitted use or reference.
    def activate(self, circuit_input_scalar: float) -> Tuple[float, np.ndarray, np.ndarray]:
        circuit_input_scalar_f64 = np.float64(circuit_input_scalar)
        internal_pre_activations = self.internal_weights * circuit_input_scalar_f64 + self.internal_biases
        internal_activations = sigmoid_numba(internal_pre_activations)
        circuit_output = np.mean(internal_activations)
        return circuit_output, internal_pre_activations, internal_activations

    def derivative_output_wrt_input(self, internal_activations: np.ndarray) -> float:
        ds_dz = sigmoid_derivative_numba(internal_activations)
        weighted_derivatives = ds_dz * self.internal_weights
        return np.mean(weighted_derivatives)

# --- ComplexFluidLearner Class (with Numba and CORRECTED derivative logic) ---
class ComplexFluidLearner:
    def __init__(self, 
                 n_inputs: int, 
                 n_outputs: int,
                 n_hidden_circuits: int = 10,
                 n_internal_units_per_circuit: int = 5,
                 learning_rate: float = 0.001,
                 seed: int = 42):
        np.random.seed(seed)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_circuits = n_hidden_circuits
        self.n_internal_units_per_circuit = n_internal_units_per_circuit # Store this
        self.learning_rate = learning_rate
        
        self.hidden_circuits_internal_weights = np.empty((n_hidden_circuits, n_internal_units_per_circuit), dtype=np.float64)
        self.hidden_circuits_internal_biases = np.empty((n_hidden_circuits, n_internal_units_per_circuit), dtype=np.float64)
        
        # We don't strictly need micro_circuits_list if all logic is in static methods
        # but it's harmless to keep for now.
        self.micro_circuits_list = [] 
        for i in range(n_hidden_circuits):
            mc = MicroCircuit(n_internal_units_per_circuit, input_scale=1.5)
            self.hidden_circuits_internal_weights[i,:] = mc.internal_weights
            self.hidden_circuits_internal_biases[i,:] = mc.internal_biases
            self.micro_circuits_list.append(mc)
            
        limit_w1 = np.sqrt(6 / (n_inputs + n_hidden_circuits))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, (n_inputs, n_hidden_circuits)).astype(np.float64)
        self.b1 = np.zeros(n_hidden_circuits, dtype=np.float64)
        
        limit_w2 = np.sqrt(6 / (n_hidden_circuits + n_outputs))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, (n_hidden_circuits, n_outputs)).astype(np.float64)
        self.b2 = np.zeros(n_outputs, dtype=np.float64)
        
        self.training_history = {'loss': [], 'val_loss': [], 'epoch': []}

    # Step 1: Modify _forward_pass_static to return internal activations
    def _forward_pass_static(x_np, W1_np, b1_np, W2_np, b2_np, 
                             hidden_circuits_iw_np, hidden_circuits_ib_np,
                             n_hidden_circuits, n_internal_units_per_circuit):
        
        hidden_circuit_inputs_linear = np.dot(x_np, W1_np) + b1_np
        hidden_circuit_outputs = np.zeros(n_hidden_circuits, dtype=np.float64)
        # Create array to store all internal activations for this sample
        all_internal_activations_sample = np.zeros((n_hidden_circuits, n_internal_units_per_circuit), dtype=np.float64)
        
        for i in range(n_hidden_circuits):
            circuit_input_scalar = np.float64(hidden_circuit_inputs_linear[i])
            internal_weights_i = hidden_circuits_iw_np[i]
            internal_biases_i = hidden_circuits_ib_np[i]
            internal_pre_activations = internal_weights_i * circuit_input_scalar + internal_biases_i
            internal_activations = sigmoid_numba(internal_pre_activations)
            hidden_circuit_outputs[i] = np.mean(internal_activations)
            all_internal_activations_sample[i, :] = internal_activations # Store them

        final_output_linear = np.dot(hidden_circuit_outputs, W2_np) + b2_np
        final_prediction = final_output_linear
        
        return final_prediction, hidden_circuit_outputs, all_internal_activations_sample # Return new cache item

    _forward_pass_jitted = staticmethod(jit(nopython=True, cache=True)(_forward_pass_static))

    # Step 3: Modify _backward_pass_static to use cached internal activations
    def _backward_pass_static(prediction_np, target_output_np, x_np, a_h_np, 
                              all_internal_activations_sample_np, # New argument
                              W1_np, W2_np, 
                              hidden_circuits_iw_np, # No need for ib_np here
                              n_hidden_circuits, n_internal_units_per_circuit): # n_internal_units not strictly needed if shape is known
        
        error_output_layer = prediction_np - target_output_np
        dW2 = np.outer(a_h_np, error_output_layer)
        db2 = error_output_layer
        error_propagated_to_hidden_outputs = np.dot(error_output_layer, W2_np.T)
        dL_dz_h = np.zeros(n_hidden_circuits, dtype=np.float64)
        
        for i in range(n_hidden_circuits):
            internal_activations_i = all_internal_activations_sample_np[i] # Use cached activations
            ds_dz = sigmoid_derivative_numba(internal_activations_i)
            weighted_derivatives = ds_dz * hidden_circuits_iw_np[i] # Use internal weights for this circuit
            circuit_derivative = np.mean(weighted_derivatives)
            dL_dz_h[i] = error_propagated_to_hidden_outputs[i] * circuit_derivative

        dW1 = np.outer(x_np, dL_dz_h)
        db1 = dL_dz_h
        return dW1, db1, dW2, db2

    _backward_pass_jitted = staticmethod(jit(nopython=True, cache=True)(_backward_pass_static))

    # Step 2: Modify forward_pass instance method to handle the new cache item
    def forward_pass(self, input_params: np.ndarray) -> Tuple[np.ndarray, Dict]:
        x_np = input_params.astype(np.float64)
        
        prediction, a_h, all_internal_acts_sample = self._forward_pass_jitted(
            x_np, self.W1, self.b1, self.W2, self.b2,
            self.hidden_circuits_internal_weights, self.hidden_circuits_internal_biases,
            self.n_hidden_circuits, self.n_internal_units_per_circuit # Pass n_internal_units
        )
        
        cache = {
            'x': x_np,
            'hidden_circuit_outputs': a_h,
            'all_internal_activations_sample': all_internal_acts_sample, # Add to cache
            'final_prediction': prediction,
        }
        return prediction, cache

    def backward_pass_to_get_grads(self, prediction: np.ndarray, target_output: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_np = cache['x'] # Already float64 from forward_pass
        a_h_np = cache['hidden_circuit_outputs']
        all_internal_acts_sample_np = cache['all_internal_activations_sample'] # Get from cache
        prediction_np = prediction # Already float64
        target_output_np = target_output.astype(np.float64)

        dW1, db1, dW2, db2 = self._backward_pass_jitted(
            prediction_np, target_output_np, x_np, a_h_np,
            all_internal_acts_sample_np, # Pass to jitted function
            self.W1, self.W2, 
            self.hidden_circuits_internal_weights, # Only iw needed for derivative
            self.n_hidden_circuits, self.n_internal_units_per_circuit
        )
        return dW1, db1, dW2, db2
    
    # --- learn, predict, plot_training_history, generate_synthetic_fluid_data, r_squared ---
    # --- These methods remain IDENTICAL to your "tunable" version that produced the output ---
    # --- as they correctly call the instance methods forward_pass and backward_pass_to_get_grads ---

    def learn( 
        self, 
        input_data: np.ndarray, 
        output_data: np.ndarray, 
        n_epochs: int = 1000,
        min_epochs_no_improve: int = 50,
        patience_no_improve: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: bool = True
    ):
        n_samples = input_data.shape[0]
        if validation_split > 0 and n_samples * validation_split >= 1:
            val_size = int(n_samples * validation_split)
            permutation = np.random.permutation(n_samples)
            val_input, val_output = input_data[permutation[:val_size]], output_data[permutation[:val_size]]
            train_input, train_output = input_data[permutation[val_size:]], output_data[permutation[val_size:]]
            n_train_samples = train_input.shape[0]
            if verbose: print(f"Training on {n_train_samples} samples, validating on {val_size} samples.")
        else:
            train_input, train_output = input_data, output_data
            val_input, val_output = None, None
            n_train_samples = train_input.shape[0]
            if verbose: print(f"Training on {n_train_samples} samples (no validation split).")


        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = None
        
        for epoch in range(n_epochs):
            epoch_train_loss = 0.0
            permutation = np.random.permutation(n_train_samples)
            shuffled_train_input = train_input[permutation]
            shuffled_train_output = train_output[permutation]
            
            for i in range(0, n_train_samples, batch_size):
                batch_input = shuffled_train_input[i:i+batch_size]
                batch_output = shuffled_train_output[i:i+batch_size]
                current_batch_size = batch_input.shape[0]
                batch_dW1, batch_db1, batch_dW2, batch_db2 = (np.zeros_like(self.W1), np.zeros_like(self.b1), 
                                                              np.zeros_like(self.W2), np.zeros_like(self.b2))
                for j in range(current_batch_size):
                    single_input, single_target = batch_input[j], batch_output[j]
                    
                    prediction, cache = self.forward_pass(single_input) 
                    
                    loss = np.mean((single_target - prediction) ** 2)
                    epoch_train_loss += loss
                    
                    dW1_s, db1_s, dW2_s, db2_s = self.backward_pass_to_get_grads(prediction, single_target, cache)
                    
                    batch_dW1 += dW1_s; batch_db1 += db1_s; batch_dW2 += dW2_s; batch_db2 += db2_s
                
                avg_dW1, avg_db1, avg_dW2, avg_db2 = (batch_dW1 / current_batch_size, batch_db1 / current_batch_size,
                                                      batch_dW2 / current_batch_size, batch_db2 / current_batch_size)
                clip_val = 1.0
                avg_dW1, avg_db1, avg_dW2, avg_db2 = (np.clip(g, -clip_val, clip_val) for g in [avg_dW1, avg_db1, avg_dW2, avg_db2])
                self.W1 -= self.learning_rate * avg_dW1; self.b1 -= self.learning_rate * avg_db1
                self.W2 -= self.learning_rate * avg_dW2; self.b2 -= self.learning_rate * avg_db2
            
            avg_epoch_train_loss = epoch_train_loss / n_train_samples
            self.training_history['loss'].append(avg_epoch_train_loss)
            self.training_history['epoch'].append(epoch)
            
            current_val_loss = avg_epoch_train_loss
            if val_input is not None:
                val_predictions, _ = self.predict(val_input, batch_size=batch_size if batch_size <= len(val_input) else len(val_input))
                current_val_loss = np.mean((val_output - val_predictions) ** 2)
            self.training_history['val_loss'].append(current_val_loss)
            
            log_interval = max(1, n_epochs // 20)
            if verbose and (epoch + 1) % log_interval == 0:
                log_msg = f"Epoch {epoch + 1:4d}: Train Loss={avg_epoch_train_loss:.6f}"
                if val_input is not None: log_msg += f", Val Loss={current_val_loss:.6f}"
                print(log_msg)
            
            loss_for_stopping = current_val_loss
            if loss_for_stopping < best_val_loss: 
                best_val_loss = loss_for_stopping
                epochs_without_improvement = 0
                best_weights = {'W1': self.W1.copy(), 'b1': self.b1.copy(), 
                                'W2': self.W2.copy(), 'b2': self.b2.copy()}
            else:
                epochs_without_improvement += 1
            
            if epoch >= min_epochs_no_improve and epochs_without_improvement >= patience_no_improve:
                if verbose: print(f"Early stopping at epoch {epoch + 1} (no improvement in val_loss).")
                if best_weights:
                    self.W1, self.b1 = best_weights['W1'], best_weights['b1']
                    self.W2, self.b2 = best_weights['W2'], best_weights['b2']
                break
        if verbose: print(f"Training done! Best validation loss: {best_val_loss:.6f}")

    def predict(self, input_data: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, None]:
        n_samples = input_data.shape[0]
        predictions = np.zeros((n_samples, self.n_outputs))
        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0:
            effective_batch_size = n_samples

        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]
                current_batch_size_pred = batch_input.shape[0]
                for j in range(current_batch_size_pred):
                    single_input = batch_input[j]
                    prediction, _ = self.forward_pass(single_input) 
                    predictions[i+j, :] = prediction
        return predictions, None

# --- Plotting Function (as before) ---
def plot_training_history(learner, title_prefix=""):
    plt.figure(figsize=(10, 6)) 
    epochs = learner.training_history['epoch']
    plt.plot(epochs, learner.training_history['loss'], label='Training Loss')
    if 'val_loss' in learner.training_history and learner.training_history['val_loss']:
         plt.plot(epochs, learner.training_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.title(f'{title_prefix}Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Synthetic Data Generation (as before) ---
def generate_synthetic_fluid_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    param1 = np.random.rand(n_samples, 1) 
    param2 = np.random.rand(n_samples, 1) - 0.5 
    output1 = 0.5 * np.sin(3 * np.pi * param1) + param2**3 + 0.2 * param1 * param2 + np.random.normal(0, 0.02, (n_samples,1))
    output2 = 0.3 * param1**2 + np.abs(param2) + 0.1 * np.cos(2 * np.pi * param1) + np.random.normal(0, 0.02, (n_samples,1))
    inputs = np.hstack((param1, param2))
    outputs = np.hstack((output1, output2))
    outputs[:,0] = (outputs[:,0] - np.min(outputs[:,0])) / (np.max(outputs[:,0]) - np.min(outputs[:,0]) + 1e-6)
    outputs[:,1] = (outputs[:,1] - np.min(outputs[:,1])) / (np.max(outputs[:,1]) - np.min(outputs[:,1]) + 1e-6)
    return inputs, outputs

# --- R-squared Metric (as before) ---
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2)
    if ss_tot == 0: 
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

# --- Main Execution (as before) ---
if __name__ == "__main__":
    print("🧠 Complex Fluid Learner - Surrogate Modeling Demo (Numba with Corrected Derivative Logic)")
    print("=" * 70) # Adjusted title

    # Using the parameters from your last run that showed poor results
    NUM_SAMPLES = 10000
    TRAIN_SPLIT_RATIO = 0.8
    VALIDATION_SPLIT_TRAIN = 0.15
    N_HIDDEN_CIRCUITS = 24 
    N_INTERNAL_UNITS = 7   
    LEARNING_RATE = 0.002    
    N_EPOCHS = 50000 # Let's keep this high, early stopping will handle it
    BATCH_SIZE = 64        
    MIN_EPOCHS_NO_IMPROVE = 150 # Increased slightly from your run
    PATIENCE_NO_IMPROVE = 300   # Increased slightly
    RANDOM_SEED_DATA = 123
    RANDOM_SEED_NETWORK = 42
    
    input_features, target_outputs = generate_synthetic_fluid_data(NUM_SAMPLES, seed=RANDOM_SEED_DATA)
    n_train_val = int(NUM_SAMPLES * TRAIN_SPLIT_RATIO)
    train_val_X, train_val_Y = input_features[:n_train_val], target_outputs[:n_train_val]
    test_X, test_Y = input_features[n_train_val:], target_outputs[n_train_val:]

    print(f"Generated {NUM_SAMPLES} samples.")
    print(f"Training/Validation data shape: X={train_val_X.shape}, Y={train_val_Y.shape}")
    print(f"Testing data shape:  X={test_X.shape}, Y={test_Y.shape}")

    n_inputs = train_val_X.shape[1]
    n_outputs = train_val_Y.shape[1]
    
    learner = ComplexFluidLearner(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden_circuits=N_HIDDEN_CIRCUITS,
        n_internal_units_per_circuit=N_INTERNAL_UNITS, # Passed to __init__
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED_NETWORK
    )
    print(f"\nInitialized ComplexFluidLearner with {learner.n_hidden_circuits} hidden circuits, "
          f"{learner.n_internal_units_per_circuit} internal units each.") # Use learner's attribute
    print(f"Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Epochs: {N_EPOCHS}")

    print("\n🚀 Starting Training...")
    import time
    start_time = time.time()
    learner.learn(
        train_val_X, train_val_Y,
        n_epochs=N_EPOCHS,
        min_epochs_no_improve=MIN_EPOCHS_NO_IMPROVE,
        patience_no_improve=PATIENCE_NO_IMPROVE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT_TRAIN, 
        verbose=True
    )
    end_time = time.time()
    print(f"--- Training finished in {end_time - start_time:.2f} seconds ---")
    
    plot_training_history(learner, f"Corrected Numba LR={LEARNING_RATE}, HC={N_HIDDEN_CIRCUITS}, IU={N_INTERNAL_UNITS} ")

    print("\n🧪 Evaluating on Test Data...")
    if test_X.shape[0] > 0:
        test_predictions, _ = learner.predict(test_X, batch_size=BATCH_SIZE)
        mse_test = np.mean((test_Y - test_predictions) ** 2)
        mae_test = np.mean(np.abs(test_Y - test_predictions))
        r2_test = r_squared(test_Y, test_predictions)
        print(f"Test Mean Squared Error (MSE): {mse_test:.6f}")
        print(f"Test Mean Absolute Error (MAE): {mae_test:.6f}")
        print(f"Test R-squared (overall):     {r2_test:.6f}")
        for i in range(n_outputs):
            r2_output_i = r_squared(test_Y[:, i], test_predictions[:, i])
            print(f"Test R-squared (Output {i+1}):   {r2_output_i:.6f}")

        output_to_plot = 0 
        plt.figure(figsize=(10, 6))
        plt.scatter(test_Y[:, output_to_plot], test_predictions[:, output_to_plot], alpha=0.5, label='Predicted vs True')
        min_val = min(np.min(test_Y[:, output_to_plot]), np.min(test_predictions[:, output_to_plot]))
        max_val = max(np.max(test_Y[:, output_to_plot]), np.max(test_predictions[:, output_to_plot]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
        plt.xlabel(f"True Output {output_to_plot+1}")
        plt.ylabel(f"Predicted Output {output_to_plot+1}")
        plt.title(f"Test Data: Predictions vs True Values for Output {output_to_plot+1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No test data to evaluate.")
    print("\n✅ Demo Completed.")
