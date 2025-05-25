import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict
from itertools import product

np.random.seed(42) # For reproducibility

def sigmoid(x):
    """Stable sigmoid activation function"""
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x_activated):
    """Derivative of sigmoid function, given activated value s = sigmoid(x)"""
    return x_activated * (1 - x_activated)

class MicroCircuit:
    """A hidden unit with internal, non-learnable micro-structure."""
    def __init__(self, n_internal_units: int = 3, input_scale: float = 1.0):
        self.n_internal_units = n_internal_units
        self.internal_weights = np.random.randn(n_internal_units) * input_scale
        self.internal_biases = np.random.randn(n_internal_units) * 0.5
        
    def activate(self, circuit_input_scalar: float) -> Tuple[float, np.ndarray, np.ndarray]:
        internal_pre_activations = self.internal_weights * circuit_input_scalar + self.internal_biases
        internal_activations = sigmoid(internal_pre_activations)
        circuit_output = np.mean(internal_activations)
        return circuit_output, internal_pre_activations, internal_activations

    def derivative_output_wrt_input(self, internal_activations: np.ndarray) -> float:
        ds_dz = sigmoid_derivative(internal_activations)
        weighted_derivatives = ds_dz * self.internal_weights
        return np.mean(weighted_derivatives)


class ComplexLearner:
    """Learner with MicroCircuits as hidden units."""
    
    def __init__(
        self, 
        n_inputs: int, # Now a parameter
        n_hidden_circuits: int = 4,
        n_internal_units_per_circuit: int = 3,
        learning_rate: float = 0.5
    ):
        self.n_inputs = n_inputs
        self.n_hidden_circuits = n_hidden_circuits
        self.learning_rate = learning_rate
        
        self.hidden_circuits = [
            MicroCircuit(n_internal_units_per_circuit, input_scale=1.5) 
            for _ in range(n_hidden_circuits)
        ]
        
        limit_w1 = np.sqrt(6 / (n_inputs + n_hidden_circuits))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, (n_inputs, n_hidden_circuits))
        self.b1 = np.zeros(n_hidden_circuits)
        
        limit_w2 = np.sqrt(6 / (n_hidden_circuits + 1))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, (n_hidden_circuits, 1))
        self.b2 = np.zeros(1)
        
        self.target_function = None
        self.training_history = {'loss': [], 'accuracy': [], 'epoch': []}
        
    def forward_pass(self, binary_input: List[int]) -> Tuple[float, Dict]:
        x = np.array(binary_input, dtype=np.float32)
        hidden_circuit_inputs_linear = np.dot(x, self.W1) + self.b1
        
        hidden_circuit_outputs = np.zeros(self.n_hidden_circuits)
        circuit_internal_activations_cache = [] 

        for i, circuit in enumerate(self.hidden_circuits):
            scalar_input_to_circuit = hidden_circuit_inputs_linear[i]
            output, _, internal_acts = circuit.activate(scalar_input_to_circuit)
            hidden_circuit_outputs[i] = output
            circuit_internal_activations_cache.append(internal_acts)
            
        final_output_linear = np.dot(hidden_circuit_outputs, self.W2) + self.b2
        final_prediction = sigmoid(final_output_linear)
        
        cache = {
            'x': x,
            'hidden_circuit_inputs_linear': hidden_circuit_inputs_linear,
            'hidden_circuit_outputs': hidden_circuit_outputs,
            'circuit_internal_activations_cache': circuit_internal_activations_cache,
            'final_output_linear': final_output_linear,
            'final_prediction': final_prediction
        }
        return float(final_prediction[0]), cache
    
    def backward_pass(self, prediction: float, target: int, cache: Dict):
        x = cache['x']
        z_h = cache['hidden_circuit_inputs_linear']
        a_h = cache['hidden_circuit_outputs']
        internal_acts_cache = cache['circuit_internal_activations_cache']
        a_o = cache['final_prediction']
        
        y = float(target)
        error_output_layer = a_o - y
        
        dW2 = np.outer(a_h, error_output_layer)
        db2 = error_output_layer
        
        error_propagated_to_hidden_outputs = np.dot(error_output_layer, self.W2.T)
        
        dL_dz_h = np.zeros_like(z_h)
        for i, circuit in enumerate(self.hidden_circuits):
            circuit_derivative = circuit.derivative_output_wrt_input(internal_acts_cache[i])
            dL_dz_h[i] = error_propagated_to_hidden_outputs[i] * circuit_derivative
            
        dW1 = np.outer(x, dL_dz_h)
        db1 = dL_dz_h
        
        clip_val = 1.0
        dW1 = np.clip(dW1, -clip_val, clip_val)
        dW2 = np.clip(dW2, -clip_val, clip_val)
        db1 = np.clip(db1, -clip_val, clip_val)
        db2 = np.clip(db2, -clip_val, clip_val)
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def evaluate_accuracy(self, all_inputs):
        correct = 0
        for binary_input in all_inputs:
            target = self.target_function(list(binary_input))
            prediction, _ = self.forward_pass(list(binary_input))
            pred_binary = 1 if prediction > 0.5 else 0
            if pred_binary == target:
                correct += 1
        return correct / len(all_inputs)

    def learn_function(
        self, 
        target_function: Callable[[List[int]], int],
        n_epochs: int = 2000,
        min_epochs: int = 100,
        patience: int = 200,
        verbose: bool = True
    ):
        self.target_function = target_function
        # Generate all_inputs based on self.n_inputs
        all_inputs = list(product([0, 1], repeat=self.n_inputs))
        
        if verbose:
            print(f"Learning function with {len(all_inputs)} examples for {self.n_inputs} inputs:")
            # Only print a few examples if there are too many
            for i, inp in enumerate(all_inputs):
                if i < 5 or len(all_inputs) <= 10 : # Print first 5 or all if <=10
                     print(f"  {inp} -> {target_function(list(inp))}")
                elif i == 5 and len(all_inputs) > 10:
                    print(f"  ... (and {len(all_inputs)-5} more examples)")
                    break

        best_accuracy = 0.0
        patience_counter = 0
        best_weights = None
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            np.random.shuffle(all_inputs)
            
            for binary_input in all_inputs:
                target = target_function(list(binary_input))
                prediction, cache = self.forward_pass(list(binary_input))
                loss = (target - prediction) ** 2
                epoch_loss += loss
                self.backward_pass(prediction, target, cache)
            
            accuracy = self.evaluate_accuracy(all_inputs)
            avg_loss = epoch_loss / len(all_inputs)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['epoch'].append(epoch)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                best_weights = {'W1': self.W1.copy(), 'b1': self.b1.copy(), 
                                'W2': self.W2.copy(), 'b2': self.b2.copy()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % (n_epochs // 10 if n_epochs >= 200 else 20) == 0 : # Log 10 times
                print(f"Epoch {epoch + 1:4d}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")
            
            if epoch >= min_epochs:
                if accuracy >= 1.0:
                    if verbose: print(f"Perfect accuracy at epoch {epoch + 1}!")
                    break
                elif patience_counter >= patience:
                    if verbose: print(f"Early stopping at epoch {epoch + 1}")
                    if best_weights:
                        self.W1, self.b1 = best_weights['W1'], best_weights['b1']
                        self.W2, self.b2 = best_weights['W2'], best_weights['b2']
                    break
        if verbose: print(f"Training done! Best accuracy: {best_accuracy:.2%}")

    def test_function(self, test_inputs: List[List[int]] = None) -> Dict:
        if test_inputs is None:
            test_inputs = [list(inp) for inp in product([0, 1], repeat=self.n_inputs)]
        
        results = {'inputs': [], 'targets': [], 'predictions': [], 
                   'binary_predictions': [], 'correct': []}
        
        print(f"\n🧪 Testing learned function ({self.n_inputs} inputs):")
        print("Input -> Target | Prediction | Binary | Correct")
        print("-" * 45)
        
        for i, binary_input in enumerate(test_inputs):
            target = self.target_function(list(binary_input))
            prediction, _ = self.forward_pass(list(binary_input))
            binary_prediction = 1 if prediction > 0.5 else 0
            correct = binary_prediction == target
            
            results['inputs'].append(binary_input)
            results['targets'].append(target)
            results['predictions'].append(prediction)
            results['binary_predictions'].append(binary_prediction)
            results['correct'].append(correct)
            
            # Print a few examples
            if i < 5 or len(test_inputs) <= 10:
                status = "✓" if correct else "✗"
                print(f"{str(list(binary_input)):12s} -> {target:1d}     | {prediction:6.3f}   | "
                      f"{binary_prediction:1d}      | {status}")
            elif i == 5 and len(test_inputs) > 10:
                print(f"  ... (and {len(test_inputs)-5} more test cases)")


        accuracy = np.mean(results['correct'])
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        return results

# --- New Harder Function Definitions ---
def parity_3_input(inputs: List[int]) -> int:
    """Output 1 if an odd number of inputs are 1."""
    return sum(inputs) % 2

def majority_3_input(inputs: List[int]) -> int:
    """Output 1 if two or more inputs are 1."""
    return 1 if sum(inputs) >= 2 else 0

def mux_4_to_1(inputs: List[int]) -> int:
    """
    4-to-1 Multiplexer.
    Inputs: [s1, s0, d3, d2, d1, d0] (6 inputs total)
    s1, s0 are select lines. d3, d2, d1, d0 are data lines.
    Output is d_i based on select lines:
    00 -> d0
    01 -> d1
    10 -> d2
    11 -> d3
    """
    s1, s0, d3, d2, d1, d0 = inputs
    if s1 == 0 and s0 == 0: return d0
    if s1 == 0 and s0 == 1: return d1
    if s1 == 1 and s0 == 0: return d2
    if s1 == 1 and s0 == 1: return d3
    return -1 # Should not happen for binary inputs

# --- Main execution and helper functions ---
def xor_function(inputs: List[int]) -> int: return inputs[0] ^ inputs[1]
def and_function(inputs: List[int]) -> int: return inputs[0] & inputs[1]
def or_function(inputs: List[int]) -> int: return inputs[0] | inputs[1]
def nand_function(inputs: List[int]) -> int: return 1 - (inputs[0] & inputs[1])

def plot_training_history(learner, func_name):
    # (Same as before)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History for {func_name} ({learner.n_inputs} inputs)")
    ax1.plot(learner.training_history['epoch'], learner.training_history['loss'])
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(learner.training_history['epoch'], learner.training_history['accuracy'])
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy')
    ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def test_single_function_complex(
    func_name: str, 
    func: Callable, 
    n_inputs: int,
    network_params: Dict, # For n_hidden_circuits, n_internal_units, lr
    training_params: Dict, # For epochs, min_epochs, patience
    verbose: bool = True
):
    max_attempts = network_params.get("max_attempts", 2)
    for attempt in range(max_attempts):
        if verbose: 
            print(f"\n🎯 Learning {func_name} ({n_inputs} inputs, attempt {attempt + 1}) with ComplexLearner...")
        
        learner = ComplexLearner(
            n_inputs=n_inputs,
            n_hidden_circuits=network_params.get("n_hidden_circuits", 5),
            n_internal_units_per_circuit=network_params.get("n_internal_units_per_circuit", 4),
            learning_rate=network_params.get("learning_rate", 0.6)
        )
        
        learner.learn_function(
            func, 
            n_epochs=training_params.get("n_epochs", 2500),
            min_epochs=training_params.get("min_epochs", 150),
            patience=training_params.get("patience", 300),
            verbose=verbose
        )
        results = learner.test_function()
        
        current_accuracy = 0.0
        if results and 'correct' in results and results['correct']:
            current_accuracy = np.mean(results['correct'])

        if current_accuracy >= 1.0:
            if verbose: print(f"✅ {func_name} learned successfully by ComplexLearner!")
            if verbose: plot_training_history(learner, func_name)
            return learner, results
        elif verbose:
            print(f"⚠️ ComplexLearner {func_name} accuracy: {current_accuracy:.1%} (attempt {attempt+1})")
    
    if verbose: print(f"❌ ComplexLearner {func_name} failed after {max_attempts} attempts")
    if verbose and 'learner' in locals(): plot_training_history(learner, func_name)
    return learner if 'learner' in locals() else None, results if 'results' in locals() else None


def main():
    print("🧠 Complex Binary Function Learner - Harder Tasks")
    print("Hidden units are MicroCircuits with internal structure.")
    print("=" * 60)
    
    # Define tasks: (name, function, n_inputs, network_params, training_params)
    tasks = [
        ("2-Input XOR", xor_function, 2, 
         {"n_hidden_circuits": 5, "n_internal_units_per_circuit": 4, "learning_rate": 0.6},
         {"n_epochs": 2000, "min_epochs": 100, "patience": 200}),
        ("3-Input Parity", parity_3_input, 3,
         {"n_hidden_circuits": 8, "n_internal_units_per_circuit": 5, "learning_rate": 0.5, "max_attempts": 3},
         {"n_epochs": 3000, "min_epochs": 200, "patience": 400}),
        ("3-Input Majority", majority_3_input, 3,
         {"n_hidden_circuits": 6, "n_internal_units_per_circuit": 4, "learning_rate": 0.6},
         {"n_epochs": 2500, "min_epochs": 150, "patience": 300}),
        ("4-to-1 MUX", mux_4_to_1, 6, # 2 select + 4 data lines
         {"n_hidden_circuits": 12, "n_internal_units_per_circuit": 5, "learning_rate": 0.4, "max_attempts": 3}, # More capacity
         {"n_epochs": 5000, "min_epochs": 500, "patience": 600}) # Longer training
    ]
    
    results_summary = {}
    
    for name, func, n_in, net_params, train_params in tasks:
        learner, results = test_single_function_complex(
            name, func, n_in, net_params, train_params, verbose=True
        ) 
        current_accuracy = 0.0
        if results and 'correct' in results and results['correct']:
             current_accuracy = np.mean(results['correct'])
        results_summary[name] = current_accuracy
    
    print(f"\n📊 Final Results Summary (ComplexLearner - Harder Tasks):")
    print("-" * 55)
    for func_name, accuracy in results_summary.items():
        status = "✅" if accuracy >= 1.0 else ("⚠️" if accuracy > 0.75 else "❌")
        print(f"{func_name:18s}: {accuracy:6.1%} {status}")
    
    overall_success = sum(1 for acc in results_summary.values() if acc >= 1.0)
    print(f"\nSuccessfully learned (100% acc): {overall_success}/{len(tasks)} functions")

if __name__ == "__main__":
    main()
