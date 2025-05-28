
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Any, Optional
from numba import jit
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE # Not strictly used
import warnings
warnings.filterwarnings('ignore')

# --- Utility Functions (ReLU, Sigmoid, BCE, kWTA, Leaky ReLU) ---
@jit(nopython=True, cache=True)
def relu_numba(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

@jit(nopython=True, cache=True)
def relu_derivative_numba(x_activated: np.ndarray) -> np.ndarray:
    return (x_activated > 0.0).astype(x_activated.dtype)

@jit(nopython=True, cache=True)
def leaky_relu_numba(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.maximum(alpha * x, x)

@jit(nopython=True, cache=True)
def leaky_relu_derivative_numba(x_activated: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    out = np.ones_like(x_activated)
    out[x_activated <= 0] = alpha
    return out

@jit(nopython=True, cache=True)
def sigmoid_numba(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))) # Added clip for stability

@jit(nopython=True, cache=True)
def binary_cross_entropy_numba_multi_task(
    y_true_binary_labels: np.ndarray,
    y_pred_logits: np.ndarray
) -> float:
    epsilon = 1e-9
    y_pred_probs = sigmoid_numba(y_pred_logits)
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1.0 - epsilon)
    if y_true_binary_labels.ndim == 1:
        bce_terms = - (y_true_binary_labels * np.log(y_pred_probs) + \
                       (1.0 - y_true_binary_labels) * np.log(1.0 - y_pred_probs))
        return np.sum(bce_terms)
    elif y_true_binary_labels.ndim == 2:
        bce_terms_batch = - (y_true_binary_labels * np.log(y_pred_probs) + \
                             (1.0 - y_true_binary_labels) * np.log(1.0 - y_pred_probs))
        return np.sum(np.sum(bce_terms_batch, axis=1)) / y_true_binary_labels.shape[0]
    else:
        return 0.0

@jit(nopython=True, cache=True)
def kwta_numba(activations: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= activations.shape[0]:
        if k <= 0: return np.zeros_like(activations)
        return activations.copy()
    result = np.zeros_like(activations)
    indices = np.argsort(activations)[-k:]
    for i in range(indices.shape[0]): result[indices[i]] = activations[indices[i]]
    return result

# --- MicroCircuit Class (MODIFIED for learnable internals and new activations) ---
class MicroCircuit:
    def __init__(self,
                 n_internal_units: int = 3,
                 input_scale: float = 1.0, # Scales initial random weights
                 aggregation: str = "mean",
                 internal_activation_type: str = "relu", # "relu" or "leaky_relu"
                 leaky_relu_alpha: float = 0.01,
                 lr_internal_weights: float = 0.001,
                 lr_internal_biases: float = 0.001,
                 internal_weight_decay: float = 0.0001,
                 initial_bias_scale: float = 0.01): # Reduced initial bias scale
        self.n_internal_units = n_internal_units
        # Internal weights are connections from the single circuit input to each internal unit
        self.internal_weights = np.random.randn(n_internal_units).astype(np.float64) * input_scale
        self.internal_biases = np.random.randn(n_internal_units).astype(np.float64) * initial_bias_scale
        self.aggregation = aggregation
        if aggregation not in ["mean", "max"]: raise ValueError("Aggregation must be 'mean' or 'max'")

        self.internal_activation_type = internal_activation_type
        self.leaky_relu_alpha = leaky_relu_alpha
        if internal_activation_type == "relu":
            self.internal_activation_fn = relu_numba
            self.internal_derivative_fn = relu_derivative_numba
        elif internal_activation_type == "leaky_relu":
            self.internal_activation_fn = lambda x: leaky_relu_numba(x, self.leaky_relu_alpha)
            self.internal_derivative_fn = lambda x: leaky_relu_derivative_numba(x, self.leaky_relu_alpha)
        else:
            raise ValueError("internal_activation_type must be 'relu' or 'leaky_relu'")

        self.lr_internal_weights = lr_internal_weights
        self.lr_internal_biases = lr_internal_biases
        self.internal_weight_decay = internal_weight_decay # For Hebbian stability

    def activate(self, circuit_input_scalar: float) -> Tuple[float, np.ndarray, np.ndarray]:
        circuit_input_scalar_f64 = np.float64(circuit_input_scalar)
        # internal_weights shape: (n_internal_units,)
        # circuit_input_scalar_f64 is scalar
        internal_pre_activations = self.internal_weights * circuit_input_scalar_f64 + self.internal_biases
        internal_activations = self.internal_activation_fn(internal_pre_activations)

        circuit_output = 0.0
        if internal_activations.size > 0:
            if self.aggregation == "mean": circuit_output = np.mean(internal_activations)
            elif self.aggregation == "max": circuit_output = np.max(internal_activations)
        return circuit_output, internal_pre_activations, internal_activations

    def update_internal_params(self, circuit_input_scalar: float, internal_activations_from_forward: np.ndarray):
        """
        Update internal weights and biases using a local, Hebbian-like rule.
        This is called *after* the global backpropagation step for W1, W2, etc.
        """
        circuit_input_scalar_f64 = np.float64(circuit_input_scalar)

        # Hebbian update for internal_weights: Δw = η * pre * post - decay * w
        # pre = circuit_input_scalar, post = internal_activations_from_forward
        delta_w_internal = self.lr_internal_weights * circuit_input_scalar_f64 * internal_activations_from_forward
        self.internal_weights += delta_w_internal
        # Apply weight decay
        if self.internal_weight_decay > 0:
            self.internal_weights -= self.lr_internal_weights * self.internal_weight_decay * self.internal_weights

        # Update for internal_biases: Δb = η_bias * post (activity-dependent)
        # This tends to increase bias for active units, decrease for inactive (if lr is negative or post can be negative)
        # Or, a simpler rule: adjust bias to encourage activity.
        # For now, a simple activity-proportional update:
        delta_b_internal = self.lr_internal_biases * internal_activations_from_forward
        self.internal_biases += delta_b_internal
        # Could add homeostatic bias adaptation here later (e.g., to target a certain mean activation)


# --- Static function for backward pass (MODIFIED to take internal weights) ---
@jit(nopython=True, cache=True)
def _backward_pass_static_jitted(
        pred_reg_np, target_reg_np,
        pred_clf_sigmoid_probs_np, target_clf_binary_labels_np,
        x_np, a_h_shared_np,
        all_internal_activations_sample_np, # Shape (n_hidden_circuits, n_internal_units_per_circuit)
        microcircuit_internal_weights_at_fwd_np, # Shape (n_hidden_circuits, n_internal_units_per_circuit)
        W1_np, W2_reg_np, W2_clf_np,
        # No hidden_circuits_iw_np directly, it's now microcircuit_internal_weights_at_fwd_np
        n_hidden_circuits, n_internal_units_per_circuit, # n_internal_units_per_circuit needed for loop
        loss_weight_reg, loss_weight_clf,
        microcircuit_aggregation_method_code: int,
        l1_activation_lambda: float,
        internal_activation_is_leaky_relu: bool, # To select derivative
        leaky_relu_alpha_static: float
    ):

    error_output_layer_reg = pred_reg_np - target_reg_np
    dW2_reg = np.outer(a_h_shared_np, error_output_layer_reg)
    db2_reg = error_output_layer_reg
    error_propagated_to_hidden_reg = np.dot(error_output_layer_reg, W2_reg_np.T) * loss_weight_reg

    error_output_layer_clf = pred_clf_sigmoid_probs_np - target_clf_binary_labels_np
    dW2_clf = np.outer(a_h_shared_np, error_output_layer_clf)
    db2_clf = error_output_layer_clf
    error_propagated_to_hidden_clf = np.dot(error_output_layer_clf, W2_clf_np.T) * loss_weight_clf

    total_error_propagated_to_hidden_outputs = error_propagated_to_hidden_reg + error_propagated_to_hidden_clf
    if l1_activation_lambda > 0:
        grad_l1_penalty = l1_activation_lambda * np.sign(a_h_shared_np)
        total_error_propagated_to_hidden_outputs += grad_l1_penalty

    dL_dz_h_shared = np.zeros(n_hidden_circuits, dtype=np.float64)
    for i in range(n_hidden_circuits):
        internal_activations_i = all_internal_activations_sample_np[i, :] # These are post-activation
        current_mc_internal_weights = microcircuit_internal_weights_at_fwd_np[i, :]

        if internal_activations_i.size > 0:
            # Derivative of internal activation function (e.g., ReLU'(z_internal) or LeakyReLU'(z_internal))
            # We have activated values, so pass them to derivative function
            if internal_activation_is_leaky_relu:
                ds_dz_internal = leaky_relu_derivative_numba(internal_activations_i, leaky_relu_alpha_static)
            else: # ReLU
                ds_dz_internal = relu_derivative_numba(internal_activations_i)

            circuit_derivative_contribution_from_units = ds_dz_internal * current_mc_internal_weights
            # This is d(internal_act)/d(circuit_input_scalar) for each unit

            circuit_derivative_wrt_circuit_input = 0.0
            if microcircuit_aggregation_method_code == 0: # MEAN
                if n_internal_units_per_circuit > 0:
                    circuit_derivative_wrt_circuit_input = np.mean(circuit_derivative_contribution_from_units)
            elif microcircuit_aggregation_method_code == 1: # MAX
                # For max, derivative is 1 for the max unit, 0 otherwise, w.r.t. that unit's activation
                # So, we need d(max_output)/d(circuit_input_scalar)
                # This is d(max_internal_act)/d(internal_pre_act_of_max_unit) * internal_weight_of_max_unit
                # Assuming internal_activations_i are the values *before* aggregation
                # The derivative of max(f(x1), f(x2)...) w.r.t. circuit_input_scalar (s_c)
                # is f'(xj) * wj where xj is the unit that produced the max output.
                idx_max = np.argmax(internal_activations_i) # Max of *activated* values
                circuit_derivative_wrt_circuit_input = circuit_derivative_contribution_from_units[idx_max]

            dL_dz_h_shared[i] = total_error_propagated_to_hidden_outputs[i] * circuit_derivative_wrt_circuit_input
        else:
            dL_dz_h_shared[i] = 0.0

    dW1 = np.outer(x_np, dL_dz_h_shared)
    db1 = dL_dz_h_shared

    return dW1, db1, dW2_reg, db2_reg, dW2_clf, db2_clf

# --- MultiTaskComplexLearner Class (MODIFIED) ---
class MultiTaskComplexLearner:
    def __init__(self,
                 n_inputs: int,
                 n_outputs_regression: int,
                 n_binary_clf_tasks: int,
                 n_hidden_circuits: int = 10,
                 n_internal_units_per_circuit: int = 5,
                 learning_rate: float = 0.001,
                 loss_weight_regression: float = 1.0,
                 loss_weight_classification: float = 1.0,
                 microcircuit_aggregation: str = "mean",
                 internal_activation_type: str = "relu", # Passed to MicroCircuit
                 leaky_relu_alpha: float = 0.01,        # Passed to MicroCircuit
                 lr_internal_weights: float = 0.0001,   # Passed to MicroCircuit
                 lr_internal_biases: float = 0.0001,    # Passed to MicroCircuit
                 internal_weight_decay: float = 0.0001, # Passed to MicroCircuit
                 l1_activation_lambda: float = 0.0,
                 use_kwta_on_circuits: bool = False,
                 kwta_k_circuits: int = 0,
                 lr_scheduler_patience: int = 50, # For main LR
                 lr_scheduler_factor: float = 0.5,  # For main LR
                 min_lr: float = 1e-7,            # For main LR
                 seed: int = 42):
        np.random.seed(seed)
        self.n_inputs = n_inputs
        self.n_outputs_regression = n_outputs_regression
        self.n_outputs_classification = n_binary_clf_tasks
        self.n_hidden_circuits = n_hidden_circuits
        self.n_internal_units_per_circuit = n_internal_units_per_circuit

        self.learning_rate_initial = learning_rate
        self.learning_rate = learning_rate
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.min_lr = min_lr
        self.epochs_lr_plateau = 0 # Counter for LR scheduler

        self.loss_weight_regression = loss_weight_regression
        self.loss_weight_classification = loss_weight_classification
        self.microcircuit_aggregation = microcircuit_aggregation
        self.microcircuit_aggregation_code = 0 if microcircuit_aggregation == "mean" else 1
        self.internal_activation_type = internal_activation_type # Store for backward pass logic
        self.leaky_relu_alpha = leaky_relu_alpha # Store for backward pass logic

        self.l1_activation_lambda = l1_activation_lambda
        self.use_kwta_on_circuits = use_kwta_on_circuits
        self.kwta_k_circuits = kwta_k_circuits if use_kwta_on_circuits else n_hidden_circuits
        if use_kwta_on_circuits and (kwta_k_circuits <= 0 or kwta_k_circuits > n_hidden_circuits) :
            print(f"Warning: kwta_k_circuits ({kwta_k_circuits}) invalid. Disabling kWTA.")
            self.use_kwta_on_circuits = False; self.kwta_k_circuits = n_hidden_circuits

        # Initialize MicroCircuits
        self.microcircuits: List[MicroCircuit] = []
        for _ in range(n_hidden_circuits):
            mc = MicroCircuit(
                n_internal_units=n_internal_units_per_circuit,
                aggregation=self.microcircuit_aggregation,
                internal_activation_type=internal_activation_type,
                leaky_relu_alpha=leaky_relu_alpha,
                lr_internal_weights=lr_internal_weights,
                lr_internal_biases=lr_internal_biases,
                internal_weight_decay=internal_weight_decay,
                input_scale=np.sqrt(2.0 / n_inputs) if n_inputs > 0 else 1.0 # He-like init for effective input to MC
            )
            self.microcircuits.append(mc)

        limit_w1 = np.sqrt(6.0 / (n_inputs + n_hidden_circuits))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, (n_inputs, n_hidden_circuits)).astype(np.float64)
        self.b1 = np.zeros(n_hidden_circuits, dtype=np.float64)

        if n_outputs_regression > 0:
            limit_w2_reg = np.sqrt(6.0 / (n_hidden_circuits + n_outputs_regression))
            self.W2_reg = np.random.uniform(-limit_w2_reg, limit_w2_reg, (n_hidden_circuits, n_outputs_regression)).astype(np.float64)
            self.b2_reg = np.zeros(n_outputs_regression, dtype=np.float64)
        else: self.W2_reg = np.empty((n_hidden_circuits, 0), dtype=np.float64); self.b2_reg = np.empty(0, dtype=np.float64)

        if self.n_outputs_classification > 0:
            limit_w2_clf = np.sqrt(6.0 / (n_hidden_circuits + self.n_outputs_classification))
            self.W2_clf = np.random.uniform(-limit_w2_clf, limit_w2_clf, (n_hidden_circuits, self.n_outputs_classification)).astype(np.float64)
            self.b2_clf = np.zeros(self.n_outputs_classification, dtype=np.float64)
        else: self.W2_clf = np.empty((n_hidden_circuits, 0), dtype=np.float64); self.b2_clf = np.empty(0, dtype=np.float64)

        self.training_history = {'loss': [], 'loss_reg': [], 'loss_clf':[], 'loss_l1_act': [], 'val_loss': [], 'val_loss_reg': [], 'val_loss_clf': [], 'epoch': [], 'learning_rate': [], 'gradient_norms_w1': [], 'gradient_norms_w2_reg': [], 'gradient_norms_w2_clf': [], 'circuit_activations_mean_epoch': [], 'weight_changes_w1': [], 'weight_changes_w2_reg': [], 'weight_changes_w2_clf': [], 'mean_internal_weights_norm_epoch': []}


    def forward_pass(self, input_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        x_np = input_params.astype(np.float64)

        # Inputs to microcircuits (linear part)
        hidden_circuit_inputs_linear = np.dot(x_np, self.W1) + self.b1 # Shape: (n_hidden_circuits,)

        a_h_shared_pre_kwta = np.zeros(self.n_hidden_circuits, dtype=np.float64)
        all_internal_activations_sample = np.zeros((self.n_hidden_circuits, self.n_internal_units_per_circuit), dtype=np.float64)
        # Store internal weights as they are at this forward pass for the backward pass
        microcircuit_internal_weights_at_fwd = np.zeros((self.n_hidden_circuits, self.n_internal_units_per_circuit), dtype=np.float64)
        microcircuit_inputs_scalars_cache = np.zeros(self.n_hidden_circuits, dtype=np.float64)


        for i in range(self.n_hidden_circuits):
            mc = self.microcircuits[i]
            circuit_input_scalar = hidden_circuit_inputs_linear[i]
            microcircuit_inputs_scalars_cache[i] = circuit_input_scalar

            circuit_output, _, internal_activations = mc.activate(circuit_input_scalar)
            # _, internal_pre_activations, internal_activations = mc.activate(circuit_input_scalar)

            a_h_shared_pre_kwta[i] = circuit_output
            all_internal_activations_sample[i, :] = internal_activations
            microcircuit_internal_weights_at_fwd[i, :] = mc.internal_weights.copy() # Crucial copy

        a_h_shared = a_h_shared_pre_kwta
        if self.use_kwta_on_circuits:
            a_h_shared = kwta_numba(a_h_shared_pre_kwta, self.kwta_k_circuits)

        prediction_reg = np.empty(self.n_outputs_regression, dtype=np.float64)
        if self.n_outputs_regression > 0:
            final_output_linear_reg = np.dot(a_h_shared, self.W2_reg) + self.b2_reg
            prediction_reg = final_output_linear_reg

        logits_clf = np.empty(self.n_outputs_classification, dtype=np.float64)
        prediction_clf_sigmoid_probs = np.empty(self.n_outputs_classification, dtype=np.float64)
        if self.n_outputs_classification > 0:
            logits_clf = np.dot(a_h_shared, self.W2_clf) + self.b2_clf
            prediction_clf_sigmoid_probs = sigmoid_numba(logits_clf)

        cache = {
            'x': x_np,
            'hidden_circuit_outputs_shared': a_h_shared, # Post-kWTA
            'all_internal_activations_sample': all_internal_activations_sample, # From MC.activate
            'microcircuit_internal_weights_at_forward': microcircuit_internal_weights_at_fwd, # For backprop
            'microcircuit_inputs_scalars': microcircuit_inputs_scalars_cache, # For MC internal learning
            'prediction_reg': prediction_reg,
            'prediction_clf_sigmoid_probs': prediction_clf_sigmoid_probs,
            'logits_clf': logits_clf,
        }
        return prediction_reg, prediction_clf_sigmoid_probs, cache

    def backward_pass_to_get_grads(self,
                                   target_reg: np.ndarray,
                                   target_clf_binary_labels: np.ndarray,
                                   cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_np = cache['x']
        a_h_shared_np = cache['hidden_circuit_outputs_shared']
        all_internal_acts_sample_np = cache['all_internal_activations_sample']
        # Use the internal weights as they were during the forward pass for gradient calculation
        microcircuit_internal_weights_at_fwd_np = cache['microcircuit_internal_weights_at_forward']
        pred_reg_np = cache['prediction_reg']
        pred_clf_sigmoid_probs_np = cache['prediction_clf_sigmoid_probs']

        target_reg_np = target_reg.astype(np.float64)
        target_clf_binary_labels_np = target_clf_binary_labels.astype(np.float64)

        eff_pred_reg_np = pred_reg_np if self.n_outputs_regression > 0 else np.empty(0,dtype=np.float64)
        eff_target_reg_np = target_reg_np if self.n_outputs_regression > 0 else np.empty(0,dtype=np.float64)
        eff_pred_clf_sigmoid_probs_np = pred_clf_sigmoid_probs_np if self.n_outputs_classification > 0 else np.empty(0,dtype=np.float64)
        eff_target_clf_binary_labels_np = target_clf_binary_labels_np if self.n_outputs_classification > 0 else np.empty(0,dtype=np.float64)

        internal_act_is_leaky = (self.internal_activation_type == "leaky_relu")

        return _backward_pass_static_jitted(
            eff_pred_reg_np, eff_target_reg_np,
            eff_pred_clf_sigmoid_probs_np, eff_target_clf_binary_labels_np,
            x_np, a_h_shared_np, all_internal_acts_sample_np,
            microcircuit_internal_weights_at_fwd_np, # Pass the snapshot of internal weights
            self.W1, self.W2_reg, self.W2_clf,
            self.n_hidden_circuits, self.n_internal_units_per_circuit,
            self.loss_weight_regression if self.n_outputs_regression > 0 else 0.0,
            self.loss_weight_classification if self.n_outputs_classification > 0 else 0.0,
            self.microcircuit_aggregation_code, self.l1_activation_lambda,
            internal_act_is_leaky, self.leaky_relu_alpha # Pass info for derivative selection
        )

    def learn( self, input_data: np.ndarray, target_data_reg: np.ndarray,
               target_data_clf_binary_labels: np.ndarray,
               n_epochs: int = 1000, min_epochs_no_improve: int = 50,
               patience_no_improve: int = 100, batch_size: int = 32,
               validation_split: float = 0.1, verbose: bool = True, clip_val: float = 1.0 ):
        n_samples = input_data.shape[0]

        if validation_split > 0 and n_samples * validation_split >= 1:
            val_size = int(n_samples * validation_split); permutation = np.random.permutation(n_samples)
            val_input = input_data[permutation[:val_size]]; val_target_reg = target_data_reg[permutation[:val_size]]
            val_target_clf_binary = target_data_clf_binary_labels[permutation[:val_size]]
            train_input = input_data[permutation[val_size:]]; train_target_reg = target_data_reg[permutation[val_size:]]
            train_target_clf_binary = target_data_clf_binary_labels[permutation[val_size:]]
            n_train_samples = train_input.shape[0]
            if verbose: print(f"Training on {n_train_samples} samples, validating on {val_size} samples.")
        else:
            train_input, train_target_reg, train_target_clf_binary = input_data, target_data_reg, target_data_clf_binary_labels
            val_input, val_target_reg, val_target_clf_binary = None, None, None
            n_train_samples = train_input.shape[0]
            if verbose: print(f"Training on {n_train_samples} samples (no validation split).")

        best_val_loss = float('inf'); epochs_without_improvement = 0; best_weights = None
        self.epochs_lr_plateau = 0 # Reset LR scheduler counter

        for epoch in range(n_epochs):
            epoch_train_loss_total,epoch_train_loss_reg,epoch_train_loss_clf,epoch_train_loss_l1_act = 0.0,0.0,0.0,0.0
            epoch_grad_norms_w1_list,epoch_grad_norms_w2_reg_list,epoch_grad_norms_w2_clf_list,epoch_circuit_activations_list = [],[],[],[]
            epoch_mean_internal_weights_norm_list = []

            prev_W1 = self.W1.copy(); prev_W2_reg = self.W2_reg.copy() if self.n_outputs_regression > 0 else None; prev_W2_clf = self.W2_clf.copy() if self.n_outputs_classification > 0 else None
            permutation = np.random.permutation(n_train_samples)
            shuffled_train_input = train_input[permutation]; shuffled_train_target_reg = train_target_reg[permutation]; shuffled_train_target_clf_binary = train_target_clf_binary[permutation]

            for i in range(0, n_train_samples, batch_size):
                batch_input = shuffled_train_input[i:i+batch_size]; batch_target_reg = shuffled_train_target_reg[i:i+batch_size]; batch_target_clf_binary_b = shuffled_train_target_clf_binary[i:i+batch_size]
                current_batch_size = batch_input.shape[0]
                batch_dW1, batch_db1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
                batch_dW2_reg = np.zeros_like(self.W2_reg) if self.n_outputs_regression > 0 else None; batch_db2_reg = np.zeros_like(self.b2_reg) if self.n_outputs_regression > 0 else None
                batch_dW2_clf = np.zeros_like(self.W2_clf) if self.n_outputs_classification > 0 else None; batch_db2_clf = np.zeros_like(self.b2_clf) if self.n_outputs_classification > 0 else None

                for j_sample in range(current_batch_size):
                    single_input = batch_input[j_sample]; single_target_reg = batch_target_reg[j_sample] if self.n_outputs_regression > 0 else np.empty(0); single_target_clf_binary_s = batch_target_clf_binary_b[j_sample] if self.n_outputs_classification > 0 else np.empty(0)

                    pred_reg, pred_clf_sigmoid_probs, cache = self.forward_pass(single_input)
                    epoch_circuit_activations_list.append(cache['hidden_circuit_outputs_shared'].copy())

                    loss_reg_sample = np.mean((single_target_reg - pred_reg) ** 2) if self.n_outputs_regression > 0 else 0.0
                    loss_clf_sample = 0.0
                    if self.n_outputs_classification > 0 and single_target_clf_binary_s.size > 0 and cache['logits_clf'].size > 0:
                        loss_clf_sample = binary_cross_entropy_numba_multi_task(single_target_clf_binary_s, cache['logits_clf'])
                    l1_act_penalty_sample = self.l1_activation_lambda * np.sum(np.abs(cache['hidden_circuit_outputs_shared'])) if self.l1_activation_lambda > 0 else 0.0
                    total_loss_sample = (self.loss_weight_regression*loss_reg_sample) + (self.loss_weight_classification*loss_clf_sample) + l1_act_penalty_sample
                    epoch_train_loss_total += total_loss_sample; epoch_train_loss_reg += loss_reg_sample; epoch_train_loss_clf += loss_clf_sample; epoch_train_loss_l1_act += l1_act_penalty_sample

                    eff_target_reg = single_target_reg if self.n_outputs_regression > 0 else np.empty((0,),dtype=np.float64)
                    eff_target_clf_binary = single_target_clf_binary_s if self.n_outputs_classification > 0 else np.empty((0,),dtype=np.float64)

                    dW1_s,db1_s,dW2_reg_s,db2_reg_s,dW2_clf_s,db2_clf_s = self.backward_pass_to_get_grads(eff_target_reg, eff_target_clf_binary, cache)

                    batch_dW1 += dW1_s; batch_db1 += db1_s
                    if self.n_outputs_regression > 0: batch_dW2_reg += dW2_reg_s; batch_db2_reg += db2_reg_s
                    if self.n_outputs_classification > 0: batch_dW2_clf += dW2_clf_s; batch_db2_clf += db2_clf_s

                    # --- Local learning for MicroCircuits (after getting global grads) ---
                    # This happens for each sample
                    current_sample_internal_weights_norms = []
                    for mc_idx in range(self.n_hidden_circuits):
                        mc_input_scalar_for_update = cache['microcircuit_inputs_scalars'][mc_idx]
                        internal_acts_for_update = cache['all_internal_activations_sample'][mc_idx, :]
                        self.microcircuits[mc_idx].update_internal_params(mc_input_scalar_for_update, internal_acts_for_update)
                        current_sample_internal_weights_norms.append(np.linalg.norm(self.microcircuits[mc_idx].internal_weights))
                    if current_sample_internal_weights_norms:
                        epoch_mean_internal_weights_norm_list.append(np.mean(current_sample_internal_weights_norms))


                # Update global weights (W1, W2) using averaged batch gradients
                avg_dW1 = batch_dW1/current_batch_size; avg_db1 = batch_db1/current_batch_size; epoch_grad_norms_w1_list.append(np.linalg.norm(avg_dW1))
                self.W1 -= self.learning_rate*np.clip(avg_dW1,-clip_val,clip_val); self.b1 -= self.learning_rate*np.clip(avg_db1,-clip_val,clip_val)
                if self.n_outputs_regression > 0 and batch_dW2_reg is not None:
                    avg_dW2_reg = batch_dW2_reg/current_batch_size; avg_db2_reg = batch_db2_reg/current_batch_size; epoch_grad_norms_w2_reg_list.append(np.linalg.norm(avg_dW2_reg))
                    self.W2_reg -= self.learning_rate*np.clip(avg_dW2_reg,-clip_val,clip_val); self.b2_reg -= self.learning_rate*np.clip(avg_db2_reg,-clip_val,clip_val)
                if self.n_outputs_classification > 0 and batch_dW2_clf is not None:
                    avg_dW2_clf = batch_dW2_clf/current_batch_size; avg_db2_clf = batch_db2_clf/current_batch_size; epoch_grad_norms_w2_clf_list.append(np.linalg.norm(avg_dW2_clf))
                    self.W2_clf -= self.learning_rate*np.clip(avg_dW2_clf,-clip_val,clip_val); self.b2_clf -= self.learning_rate*np.clip(avg_db2_clf,-clip_val,clip_val)

            # --- End of Epoch ---
            weight_change_w1 = np.linalg.norm(self.W1-prev_W1); weight_change_w2_reg = np.linalg.norm(self.W2_reg-prev_W2_reg) if self.n_outputs_regression > 0 and prev_W2_reg is not None else 0.0; weight_change_w2_clf = np.linalg.norm(self.W2_clf-prev_W2_clf) if self.n_outputs_classification > 0 and prev_W2_clf is not None else 0.0
            avg_epoch_train_loss = epoch_train_loss_total/n_train_samples if n_train_samples > 0 else 0; avg_epoch_train_loss_reg = epoch_train_loss_reg/n_train_samples if n_train_samples > 0 else 0; avg_epoch_train_loss_clf = epoch_train_loss_clf/n_train_samples if n_train_samples > 0 else 0; avg_epoch_train_loss_l1_act = epoch_train_loss_l1_act/n_train_samples if n_train_samples > 0 else 0
            self.training_history['loss'].append(avg_epoch_train_loss); self.training_history['loss_reg'].append(avg_epoch_train_loss_reg); self.training_history['loss_clf'].append(avg_epoch_train_loss_clf); self.training_history['loss_l1_act'].append(avg_epoch_train_loss_l1_act); self.training_history['epoch'].append(epoch); self.training_history['learning_rate'].append(self.learning_rate)
            self.training_history['gradient_norms_w1'].append(np.mean(epoch_grad_norms_w1_list) if epoch_grad_norms_w1_list else 0); self.training_history['gradient_norms_w2_reg'].append(np.mean(epoch_grad_norms_w2_reg_list) if epoch_grad_norms_w2_reg_list else 0); self.training_history['gradient_norms_w2_clf'].append(np.mean(epoch_grad_norms_w2_clf_list) if epoch_grad_norms_w2_clf_list else 0)
            self.training_history['circuit_activations_mean_epoch'].append(np.mean(epoch_circuit_activations_list,axis=0) if epoch_circuit_activations_list else np.zeros(self.n_hidden_circuits)); self.training_history['weight_changes_w1'].append(weight_change_w1); self.training_history['weight_changes_w2_reg'].append(weight_change_w2_reg); self.training_history['weight_changes_w2_clf'].append(weight_change_w2_clf)
            self.training_history['mean_internal_weights_norm_epoch'].append(np.mean(epoch_mean_internal_weights_norm_list) if epoch_mean_internal_weights_norm_list else 0)


            current_val_loss_total = avg_epoch_train_loss; current_val_loss_reg = avg_epoch_train_loss_reg; current_val_loss_clf = avg_epoch_train_loss_clf
            if val_input is not None and len(val_input) > 0:
                val_preds_reg, val_preds_clf_sigmoid_probs, val_logits_clf_val = self.predict(val_input, batch_size=batch_size)
                current_val_loss_reg = np.mean((val_target_reg - val_preds_reg)**2) if self.n_outputs_regression > 0 else 0.0
                current_val_loss_clf = 0.0
                if self.n_outputs_classification > 0 and val_target_clf_binary.size > 0 and val_logits_clf_val.size > 0:
                     current_val_loss_clf = binary_cross_entropy_numba_multi_task(val_target_clf_binary, val_logits_clf_val)
                current_val_loss_total = (self.loss_weight_regression*current_val_loss_reg) + (self.loss_weight_classification*current_val_loss_clf)

            self.training_history['val_loss'].append(current_val_loss_total); self.training_history['val_loss_reg'].append(current_val_loss_reg); self.training_history['val_loss_clf'].append(current_val_loss_clf)
            log_interval = max(1, n_epochs//20) if n_epochs > 0 else 1
            if verbose and (epoch+1)%log_interval == 0: print(f"Epoch {epoch+1:4d}/{n_epochs}: Train Loss={avg_epoch_train_loss:.4f} (Reg: {avg_epoch_train_loss_reg:.4f}, Clf: {avg_epoch_train_loss_clf:.4f}), Val Loss={current_val_loss_total:.4f} (Reg: {current_val_loss_reg:.4f}, Clf: {current_val_loss_clf:.4f}), LR={self.learning_rate:.2e}")

            loss_for_stopping_and_lr = current_val_loss_total
            if loss_for_stopping_and_lr < best_val_loss:
                best_val_loss = loss_for_stopping_and_lr; epochs_without_improvement = 0; self.epochs_lr_plateau = 0
                best_weights = {'W1':self.W1.copy(),'b1':self.b1.copy()}
                if self.n_outputs_regression > 0: best_weights['W2_reg']=self.W2_reg.copy(); best_weights['b2_reg']=self.b2_reg.copy()
                if self.n_outputs_classification > 0: best_weights['W2_clf']=self.W2_clf.copy(); best_weights['b2_clf']=self.b2_clf.copy()
                # Could also save best microcircuit internal weights, but they learn continuously
            else:
                epochs_without_improvement += 1
                self.epochs_lr_plateau +=1

            # LR Scheduler
            if self.epochs_lr_plateau >= self.lr_scheduler_patience and self.learning_rate > self.min_lr:
                new_lr = self.learning_rate * self.lr_scheduler_factor
                self.learning_rate = max(new_lr, self.min_lr)
                if verbose: print(f"Reducing learning rate to {self.learning_rate:.2e} at epoch {epoch+1}")
                self.epochs_lr_plateau = 0 # Reset counter after reduction

            if epoch >= min_epochs_no_improve and epochs_without_improvement >= patience_no_improve:
                if verbose: print(f"Early stopping at epoch {epoch+1}.")
                if best_weights: self.W1,self.b1=best_weights['W1'],best_weights['b1']
                if self.n_outputs_regression > 0 and 'W2_reg' in best_weights: self.W2_reg,self.b2_reg=best_weights['W2_reg'],best_weights['b2_reg']
                if self.n_outputs_classification > 0 and 'W2_clf' in best_weights: self.W2_clf,self.b2_clf=best_weights['W2_clf'],best_weights['b2_clf']
                break
        if verbose: print(f"Training done! Best validation loss: {best_val_loss:.6f}")


    def predict(self, input_data: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = input_data.shape[0]
        predictions_reg = np.zeros((n_samples, self.n_outputs_regression)) if self.n_outputs_regression > 0 else np.empty((n_samples, 0))
        predictions_clf_sigmoid_probs = np.zeros((n_samples, self.n_outputs_classification)) if self.n_outputs_classification > 0 else np.empty((n_samples, 0))
        logits_clf_all = np.zeros((n_samples, self.n_outputs_classification)) if self.n_outputs_classification > 0 else np.empty((n_samples, 0))

        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0: effective_batch_size = n_samples
        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]; current_batch_size_pred = batch_input.shape[0]
                for j_sample in range(current_batch_size_pred):
                    single_input = batch_input[j_sample]; pred_reg_s, pred_clf_sigmoid_probs_s, cache_s = self.forward_pass(single_input)
                    if self.n_outputs_regression > 0: predictions_reg[i+j_sample, :] = pred_reg_s
                    if self.n_outputs_classification > 0:
                        predictions_clf_sigmoid_probs[i+j_sample, :] = pred_clf_sigmoid_probs_s
                        logits_clf_all[i+j_sample, :] = cache_s['logits_clf']
        return predictions_reg, predictions_clf_sigmoid_probs, logits_clf_all

    def get_circuit_activations(self, input_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        n_samples = input_data.shape[0]; activations = np.zeros((n_samples, self.n_hidden_circuits))
        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0: effective_batch_size = n_samples
        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]; current_bs = batch_input.shape[0]
                for j_sample in range(current_bs):
                    single_input = batch_input[j_sample]; _, _, cache = self.forward_pass(single_input)
                    activations[i+j_sample] = cache['hidden_circuit_outputs_shared']
        return activations

    def get_microcircuit_internal_weights_norms(self) -> List[float]:
        return [np.linalg.norm(mc.internal_weights) for mc in self.microcircuits]

    def get_microcircuit_internal_biases_means(self) -> List[float]:
        return [np.mean(mc.internal_biases) for mc in self.microcircuits]


# --- Plotting Functions (MODIFIED for new history item) ---
def plot_comprehensive_analysis(learner, test_X, test_Y_reg, test_Y_clf_binary_labels, num_binary_clf_tasks, title_prefix=""):
    fig = plt.figure(figsize=(22, 30)); epochs = learner.training_history['epoch'] # Increased height for new plot
    # Row 1: Losses
    ax1 = plt.subplot(8,3,1); plt.plot(epochs, learner.training_history['loss'], label='Total Training Loss', linewidth=2)
    if learner.training_history['val_loss'] and len(learner.training_history['val_loss'])==len(epochs): plt.plot(epochs, learner.training_history['val_loss'], label='Total Validation Loss', linewidth=2, linestyle='--')
    plt.ylabel('Total Loss'); plt.title('Total Loss History'); plt.legend(); plt.grid(True, alpha=0.3)
    ax2 = plt.subplot(8,3,2)
    if epochs and learner.n_outputs_regression > 0: plt.plot(epochs, learner.training_history['loss_reg'], label='Regression Training', linewidth=2)
    if learner.training_history['val_loss_reg'] and len(learner.training_history['val_loss_reg'])==len(epochs) and learner.n_outputs_regression > 0: plt.plot(epochs, learner.training_history['val_loss_reg'], label='Regression Validation', linewidth=2, linestyle='--')
    plt.ylabel('Regression Loss (MSE)'); plt.title('Regression Loss History'); plt.legend(); plt.grid(True, alpha=0.3)
    ax3 = plt.subplot(8,3,3)
    if epochs and learner.n_outputs_classification > 0: plt.plot(epochs, learner.training_history['loss_clf'], label='Classification Training (BCE Sum)', linewidth=2)
    if learner.training_history['val_loss_clf'] and len(learner.training_history['val_loss_clf'])==len(epochs) and learner.n_outputs_classification > 0: plt.plot(epochs, learner.training_history['val_loss_clf'], label='Classification Validation (BCE Sum)', linewidth=2, linestyle='--')
    plt.ylabel('Classification Loss (BCE Sum)'); plt.title('Classification Loss History'); plt.legend(); plt.grid(True, alpha=0.3)

    # Row 2: Gradients, Weight Changes, Mean Circuit Activations
    ax4 = plt.subplot(8,3,4)
    if epochs:
        if learner.training_history['gradient_norms_w1']: plt.plot(epochs, learner.training_history['gradient_norms_w1'], label='Grad Norm W1', linewidth=1.5, alpha=0.8)
        if learner.n_outputs_regression > 0 and learner.training_history['gradient_norms_w2_reg']: plt.plot(epochs, learner.training_history['gradient_norms_w2_reg'], label='Grad Norm W2_Reg', linewidth=1.5, alpha=0.8)
        if learner.n_outputs_classification > 0 and learner.training_history['gradient_norms_w2_clf']: plt.plot(epochs, learner.training_history['gradient_norms_w2_clf'], label='Grad Norm W2_Clf', linewidth=1.5, alpha=0.8)
    plt.ylabel('Gradient Norm'); plt.title('Gradient Norm Evolution (Global)'); plt.legend(); plt.grid(True, alpha=0.3)
    if any(learner.training_history['gradient_norms_w1']) or \
       (learner.n_outputs_regression > 0 and any(learner.training_history['gradient_norms_w2_reg'])) or \
       (learner.n_outputs_classification > 0 and any(learner.training_history['gradient_norms_w2_clf'])):
        plt.yscale('log', nonpositive='clip')

    ax5 = plt.subplot(8,3,5)
    if epochs:
        if learner.training_history['weight_changes_w1']: plt.plot(epochs, learner.training_history['weight_changes_w1'], label='Weight Change W1', linewidth=1.5, alpha=0.8)
        if learner.n_outputs_regression > 0 and learner.training_history['weight_changes_w2_reg']: plt.plot(epochs, learner.training_history['weight_changes_w2_reg'], label='Weight Change W2_Reg', linewidth=1.5, alpha=0.8)
        if learner.n_outputs_classification > 0 and learner.training_history['weight_changes_w2_clf']: plt.plot(epochs, learner.training_history['weight_changes_w2_clf'], label='Weight Change W2_Clf', linewidth=1.5, alpha=0.8)
    plt.ylabel('Weight Change Magnitude'); plt.title('Weight Change Evolution (Global)'); plt.legend(); plt.grid(True, alpha=0.3)
    if any(learner.training_history['weight_changes_w1']) or \
       (learner.n_outputs_regression > 0 and any(learner.training_history['weight_changes_w2_reg'])) or \
       (learner.n_outputs_classification > 0 and any(learner.training_history['weight_changes_w2_clf'])):
        plt.yscale('log', nonpositive='clip')

    ax6 = plt.subplot(8,3,6); circuit_activations_epoch_mean = np.array(learner.training_history['circuit_activations_mean_epoch'])
    if circuit_activations_epoch_mean.size > 0 and circuit_activations_epoch_mean.ndim == 2:
        im = plt.imshow(circuit_activations_epoch_mean.T, aspect='auto', cmap='viridis', interpolation='nearest'); plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04); plt.ylabel('Circuit Index'); plt.xlabel('Epoch'); plt.title('Mean Circuit Outputs (Epoch)')

    # Row 3: Regression Performance
    ax7 = plt.subplot(8,3,7)
    if test_X.shape[0] > 0 and learner.n_outputs_regression > 0:
        test_preds_reg, _, _ = learner.predict(test_X)
        if test_Y_reg.size > 0 and test_preds_reg.size > 0:
            task_idx_reg_plot = 0
            plt.scatter(test_Y_reg[:, task_idx_reg_plot].flatten(), test_preds_reg[:, task_idx_reg_plot].flatten(), alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
            min_val = min(np.min(test_Y_reg[:, task_idx_reg_plot]), np.min(test_preds_reg[:, task_idx_reg_plot])); max_val = max(np.max(test_Y_reg[:, task_idx_reg_plot]), np.max(test_preds_reg[:, task_idx_reg_plot]))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit'); plt.xlabel(f'True Values (Reg Task {task_idx_reg_plot+1})'); plt.ylabel(f'Predicted Values (Reg Task {task_idx_reg_plot+1})'); plt.title(f'Regression Task {task_idx_reg_plot+1}: Predicted vs True'); plt.legend()
        else: ax7.text(0.5,0.5,"No regression data.",ha='center',va='center',transform=ax7.transAxes)
    else: ax7.text(0.5,0.5,"No regression task or test data.",ha='center',va='center',transform=ax7.transAxes)
    plt.grid(True,alpha=0.3)

    ax8 = plt.subplot(8,3,8)
    if test_X.shape[0] > 0 and learner.n_outputs_regression > 0:
        test_preds_reg, _, _ = learner.predict(test_X) # Re-predict or use from above
        if test_Y_reg.size > 0 and test_preds_reg.size > 0:
            task_idx_reg_plot = 0; residuals = test_Y_reg[:,task_idx_reg_plot].flatten() - test_preds_reg[:,task_idx_reg_plot].flatten()
            plt.scatter(test_preds_reg[:,task_idx_reg_plot].flatten(), residuals, alpha=0.5, s=15, edgecolors='k', linewidths=0.5); plt.axhline(y=0, color='r', linestyle='--'); plt.xlabel(f'Predicted Values (Reg Task {task_idx_reg_plot+1})'); plt.ylabel('Residuals'); plt.title(f'Residual Analysis (Reg Task {task_idx_reg_plot+1})')
        else: ax8.text(0.5,0.5,"No regression data.",ha='center',va='center',transform=ax8.transAxes)
    else: ax8.text(0.5,0.5,"No regression task or test data.",ha='center',va='center',transform=ax8.transAxes)
    plt.grid(True,alpha=0.3)

    ax9 = plt.subplot(8,3,9)
    if test_X.shape[0] > 0 and learner.n_outputs_regression > 0:
        test_preds_reg, _, _ = learner.predict(test_X) # Re-predict or use from above
        if test_Y_reg.size > 0 and test_preds_reg.size > 0:
            task_idx_reg_plot = 0; residuals = test_Y_reg[:,task_idx_reg_plot].flatten() - test_preds_reg[:,task_idx_reg_plot].flatten()
            sns.histplot(residuals, bins=30, kde=True, ax=ax9, color='skyblue'); plt.xlabel('Residuals'); plt.ylabel('Density'); plt.title(f'Residual Distribution (Reg Task {task_idx_reg_plot+1})')
        else: ax9.text(0.5,0.5,"No regression data.",ha='center',va='center',transform=ax9.transAxes)
    else: ax9.text(0.5,0.5,"No regression task or test data.",ha='center',va='center',transform=ax9.transAxes)
    plt.grid(True,alpha=0.3)

    # Row 4: Circuit Activations on Test Data
    ax10 = plt.subplot(8,3,10)
    if test_X.shape[0] > 0:
        sample_size_for_activation_analysis = min(500, test_X.shape[0]); circuit_activations_test = learner.get_circuit_activations(test_X[:sample_size_for_activation_analysis])
        if circuit_activations_test.size > 0: im = plt.imshow(circuit_activations_test.T, aspect='auto', cmap='plasma', interpolation='nearest'); plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04); plt.ylabel('Circuit Index'); plt.xlabel(f'Sample Index (first {sample_size_for_activation_analysis})'); plt.title('Circuit Outputs (Test Data)')
        else: ax10.text(0.5,0.5,"No circuit activations.",ha='center',va='center',transform=ax10.transAxes)
    else: ax10.text(0.5,0.5,"No test data.",ha='center',va='center',transform=ax10.transAxes)

    ax11 = plt.subplot(8,3,11)
    if test_X.shape[0] > 0 and 'circuit_activations_test' in locals() and circuit_activations_test.size > 0 and circuit_activations_test.shape[1] > 1:
        circuit_corr = np.corrcoef(circuit_activations_test.T); im = plt.imshow(circuit_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest'); plt.colorbar(im, ax=ax11, fraction=0.046, pad=0.04); plt.xlabel('Circuit Index'); plt.ylabel('Circuit Index'); plt.title('Circuit Output Correlation')
    else: ax11.text(0.5,0.5,"Not enough data/circuits for corr.",ha='center',va='center',transform=ax11.transAxes)

    ax12 = plt.subplot(8,3,12)
    if test_X.shape[0] > 0 and 'circuit_activations_test' in locals() and circuit_activations_test.size > 0:
        circuit_means = np.mean(circuit_activations_test, axis=0); circuit_stds = np.std(circuit_activations_test, axis=0); x_pos = np.arange(len(circuit_means)); plt.bar(x_pos, circuit_means, yerr=circuit_stds, alpha=0.7, capsize=3, color='teal'); plt.xlabel('Circuit Index'); plt.ylabel('Mean Output ± Std'); plt.title('Circuit Output Statistics');
        if 'x_pos' in locals() and len(x_pos) > 0: plt.xticks(x_pos[::max(1, len(x_pos)//10)])
    else: ax12.text(0.5,0.5,"No circuit stats.",ha='center',va='center',transform=ax12.transAxes)

    # Row 5: PCA
    ax13 = plt.subplot(8,3,13) # Input PCA
    if test_X.shape[0] > 0:
        sample_size_for_pca = min(1000, test_X.shape[0]); test_X_sample = test_X[:sample_size_for_pca]
        test_Y_clf_labels_sample_for_pca = test_Y_clf_binary_labels[:sample_size_for_pca, 0] if num_binary_clf_tasks > 0 and test_Y_clf_binary_labels.ndim == 2 and test_Y_clf_binary_labels.shape[1] > 0 else None
        if test_X_sample.shape[1] >= 2:
            pca_input = PCA(n_components=2); X_pca = pca_input.fit_transform(test_X_sample)
            cmap_input = 'viridis' if num_binary_clf_tasks == 0 or test_Y_clf_labels_sample_for_pca is None else 'tab10'
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test_Y_clf_labels_sample_for_pca, cmap=cmap_input, alpha=0.6, s=20)
            if num_binary_clf_tasks > 0 and test_Y_clf_labels_sample_for_pca is not None: plt.colorbar(scatter, ax=ax13, fraction=0.046, pad=0.04, label="Task C1 Label")
            plt.xlabel(f'PC1 ({pca_input.explained_variance_ratio_[0]:.1%} var)'); plt.ylabel(f'PC2 ({pca_input.explained_variance_ratio_[1]:.1%} var)'); plt.title('Input Space (PCA)')
        else: ax13.text(0.5,0.5,"Need >=2 input features for PCA.",ha='center',va='center',transform=ax13.transAxes)
    else: ax13.text(0.5,0.5,"No test data for PCA.",ha='center',va='center',transform=ax13.transAxes)

    ax14 = plt.subplot(8,3,14) # Hidden PCA
    if test_X.shape[0] > 0 and 'circuit_activations_test' in locals() and circuit_activations_test.size > 0: # Use circuit_activations_test from ax10
        circuit_activations_test_sample = learner.get_circuit_activations(test_X_sample) # Or re-get for consistent sample
        if circuit_activations_test_sample.shape[1] >= 2:
            pca_hidden = PCA(n_components=2); hidden_pca = pca_hidden.fit_transform(circuit_activations_test_sample)
            cmap_hidden = 'viridis' if num_binary_clf_tasks == 0 or test_Y_clf_labels_sample_for_pca is None else 'tab10'
            scatter = plt.scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=test_Y_clf_labels_sample_for_pca, cmap=cmap_hidden, alpha=0.6, s=20)
            if num_binary_clf_tasks > 0 and test_Y_clf_labels_sample_for_pca is not None: plt.colorbar(scatter, ax=ax14, fraction=0.046, pad=0.04, label="Task C1 Label")
            plt.xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.1%} var)'); plt.ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.1%} var)'); plt.title('Hidden Rep (Circuit Outputs PCA)')
        else: ax14.text(0.5,0.5,"Need >=2 circuits for PCA.",ha='center',va='center',transform=ax14.transAxes)
    else: ax14.text(0.5,0.5,"No hidden activations for PCA.",ha='center',va='center',transform=ax14.transAxes)

    ax15 = plt.subplot(8,3,15) # Decision Boundary placeholder
    ax15.text(0.5,0.5,"Decision boundary plot requires\n2D input & single classification task\n(or modified plotting for multi-task).",ha='center',va='center',transform=ax15.transAxes)

    # Row 6: Performance Summary, LR, CM
    ax16 = plt.subplot(8,3,16)
    if test_X.shape[0] > 0:
        test_preds_reg, test_preds_clf_sigmoid_probs, _ = learner.predict(test_X)
        metrics_list = []; values_list = [];
        # Determine number of colors needed
        num_metrics_total = 0
        if learner.n_outputs_regression > 0 and test_Y_reg.ndim == 2: num_metrics_total += 2 * learner.n_outputs_regression
        if num_binary_clf_tasks > 0 and test_Y_clf_binary_labels.ndim == 2: num_metrics_total += num_binary_clf_tasks
        colors_list_cmap = plt.cm.get_cmap('viridis', num_metrics_total if num_metrics_total > 0 else 1)
        color_idx = 0

        for task_i in range(learner.n_outputs_regression):
            if test_Y_reg.ndim == 2 and test_Y_reg.shape[1] > task_i and test_preds_reg.shape[1] > task_i:
                mse = np.mean((test_Y_reg[:, task_i] - test_preds_reg[:, task_i]) ** 2)
                r2 = r_squared(test_Y_reg[:, task_i], test_preds_reg[:, task_i])
                metrics_list.extend([f'MSE (R{task_i+1})', f'R² (R{task_i+1})']); values_list.extend([mse, r2])
        for task_i in range(num_binary_clf_tasks):
            if test_Y_clf_binary_labels.ndim == 2 and test_Y_clf_binary_labels.shape[1] > task_i and test_preds_clf_sigmoid_probs.shape[1] > task_i:
                true_labels_task_i = test_Y_clf_binary_labels[:, task_i].flatten()
                pred_labels_task_i = (test_preds_clf_sigmoid_probs[:, task_i] > 0.5).astype(int)
                acc = np.mean(true_labels_task_i == pred_labels_task_i)
                metrics_list.append(f'Acc (C{task_i+1})'); values_list.append(acc)
        if metrics_list:
            bars = plt.bar(metrics_list, values_list, color=[colors_list_cmap(i) for i in range(len(values_list))], alpha=0.75)
            plt.ylabel('Metric Value'); plt.title('Performance Summary (Test)'); plt.xticks(rotation=45, ha="right", fontsize=8)
            for bar_idx, bar in enumerate(bars): yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values_list, default=1), f'{yval:.3f}', ha='center', va='bottom', fontsize=7)
        else: ax16.text(0.5,0.5,"No tasks with data for metrics.",ha='center',va='center',transform=ax16.transAxes)
    else: ax16.text(0.5,0.5,"No test data for metrics.",ha='center',va='center',transform=ax16.transAxes)

    ax17 = plt.subplot(8,3,17) # Learning Rate
    if epochs and learner.training_history['learning_rate']: plt.plot(epochs, learner.training_history['learning_rate'], 'g-', linewidth=2); plt.ylabel('Learning Rate'); plt.xlabel('Epoch'); plt.title('Learning Rate Schedule (Global)'); plt.grid(True, alpha=0.3)
    else: ax17.text(0.5,0.5,"No LR history.",ha='center',va='center',transform=ax17.transAxes)

    ax18 = plt.subplot(8,3,18) # Confusion Matrix
    if num_binary_clf_tasks > 0 and test_X.shape[0] > 0 and test_Y_clf_binary_labels.size > 0 and 'test_preds_clf_sigmoid_probs' in locals() and test_preds_clf_sigmoid_probs.size > 0:
        try:
            from sklearn.metrics import confusion_matrix
            task_idx_cm = 0
            if test_Y_clf_binary_labels.ndim == 2 and test_Y_clf_binary_labels.shape[1] > task_idx_cm and \
               test_preds_clf_sigmoid_probs.ndim == 2 and test_preds_clf_sigmoid_probs.shape[1] > task_idx_cm:
                true_labels_cm = test_Y_clf_binary_labels[:, task_idx_cm].flatten()
                pred_labels_cm = (test_preds_clf_sigmoid_probs[:, task_idx_cm] > 0.5).astype(int)
                cm = confusion_matrix(true_labels_cm, pred_labels_cm, labels=[0,1])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax18, xticklabels=["C0","C1"], yticklabels=["C0","C1"])
                plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title(f"CM (Binary Task C{task_idx_cm+1})")
            else: ax18.text(0.5,0.5,"Data shape error for CM.",ha='center',va='center',transform=ax18.transAxes)
        except ImportError: ax18.text(0.5,0.5,"sklearn not found for CM.",ha='center',va='center',transform=ax18.transAxes)
    else: ax18.text(0.5,0.5,"No classification task or data for CM.",ha='center',va='center',transform=ax18.transAxes)

    # Row 7: Microcircuit Internal Weight Norms
    ax19 = plt.subplot(8,3,19)
    if epochs and learner.training_history['mean_internal_weights_norm_epoch']:
        plt.plot(epochs, learner.training_history['mean_internal_weights_norm_epoch'], label='Mean MC Internal Weight Norm', linewidth=2, color='purple')
        plt.ylabel('Avg. L2 Norm'); plt.xlabel('Epoch'); plt.title('MC Internal Weight Norm Evolution'); plt.legend(); plt.grid(True, alpha=0.3)
        if any(learner.training_history['mean_internal_weights_norm_epoch']): plt.yscale('log', nonpositive='clip')
    else: ax19.text(0.5,0.5,"No MC internal weight history.",ha='center',va='center',transform=ax19.transAxes)

    ax20 = plt.subplot(8,3,20) # Distribution of MC internal weights at end
    final_internal_weights = []
    for mc in learner.microcircuits: final_internal_weights.extend(mc.internal_weights)
    if final_internal_weights:
        sns.histplot(final_internal_weights, bins=30, kde=True, ax=ax20, color='darkmagenta')
        plt.xlabel('MC Internal Weight Value'); plt.ylabel('Density'); plt.title('Final MC Internal Weight Dist.')
    else: ax20.text(0.5,0.5,"No MC internal weights.",ha='center',va='center',transform=ax20.transAxes)

    ax21 = plt.subplot(8,3,21) # Distribution of MC internal biases at end
    final_internal_biases = []
    for mc in learner.microcircuits: final_internal_biases.extend(mc.internal_biases)
    if final_internal_biases:
        sns.histplot(final_internal_biases, bins=30, kde=True, ax=ax21, color='indigo')
        plt.xlabel('MC Internal Bias Value'); plt.ylabel('Density'); plt.title('Final MC Internal Bias Dist.')
    else: ax21.text(0.5,0.5,"No MC internal biases.",ha='center',va='center',transform=ax21.transAxes)


    plt.suptitle(f'{title_prefix}Comprehensive Neural Network Analysis', fontsize=20, y=0.99); plt.tight_layout(rect=[0, 0.02, 1, 0.975]); plt.show()

# --- Plotting Brain-like and Learning Dynamics (largely same, ensure they run) ---
def plot_brain_like_analysis(learner, test_X, test_Y_reg, test_Y_clf_binary_labels, num_binary_clf_tasks):
    if test_X.shape[0] == 0: print("Skipping brain-like analysis: No test data."); return
    fig, axes = plt.subplots(2,3,figsize=(18,11)); fig.subplots_adjust(hspace=0.4,wspace=0.3)
    sample_size_analysis = min(500, test_X.shape[0])
    test_X_subset = test_X[:sample_size_analysis]
    test_Y_clf_labels_subset_for_selectivity = None
    if num_binary_clf_tasks > 0 and test_Y_clf_binary_labels.ndim == 2 and test_Y_clf_binary_labels.shape[0] >= sample_size_analysis and test_Y_clf_binary_labels.shape[1] > 0:
        test_Y_clf_labels_subset_for_selectivity = test_Y_clf_binary_labels[:sample_size_analysis, 0]

    circuit_activations = learner.get_circuit_activations(test_X_subset)
    if circuit_activations.size == 0: print("Skipping brain-like analysis: No circuit activations obtained."); return

    ax = axes[0,0]; sparsity_levels = []
    if circuit_activations.shape[1] > 0:
        threshold_sparsity = 1e-5
        for i in range(circuit_activations.shape[1]): sparsity = np.mean(np.abs(circuit_activations[:,i]) < threshold_sparsity); sparsity_levels.append(sparsity)
        if sparsity_levels: ax.bar(range(len(sparsity_levels)), sparsity_levels, alpha=0.7, color='skyblue'); ax.set_xlabel('Circuit Index'); ax.set_ylabel(f'Sparsity Level (<{threshold_sparsity} act.)'); ax.set_title('Circuit Output Sparsity'); ax.grid(True,alpha=0.3); ax.set_ylim(0,1)
    else: ax.text(0.5,0.5,"No circuits for sparsity analysis.",ha='center',va='center',transform=ax.transAxes)

    ax = axes[0,1]; selectivity_scores = []
    if num_binary_clf_tasks > 0 and test_Y_clf_labels_subset_for_selectivity is not None and circuit_activations.shape[1] > 0:
        unique_labels = np.unique(test_Y_clf_labels_subset_for_selectivity)
        if len(unique_labels) > 1: # Need at least two classes for selectivity
            for i in range(circuit_activations.shape[1]):
                class_responses = []
                for class_idx in unique_labels:
                    mask = test_Y_clf_labels_subset_for_selectivity.flatten() == class_idx
                    if np.sum(mask) > 0: mean_response = np.mean(circuit_activations[mask,i]); class_responses.append(mean_response)
                if len(class_responses) > 1 and (np.abs(np.mean(class_responses)) > 1e-8 or np.std(class_responses) > 1e-8): selectivity = np.std(class_responses) / (np.abs(np.mean(class_responses)) + 1e-8); selectivity_scores.append(selectivity)
                else: selectivity_scores.append(0)
            if selectivity_scores: ax.bar(range(len(selectivity_scores)), selectivity_scores, alpha=0.7, color='lightcoral'); ax.set_xlabel('Circuit Index'); ax.set_ylabel('Selectivity (CV to Task C1 classes)'); ax.set_title('Circuit Output Selectivity (Task C1)'); ax.grid(True,alpha=0.3)
        else: ax.text(0.5,0.5,"Need >1 class for selectivity.",ha='center',va='center',transform=ax.transAxes)
    else: ax.text(0.5,0.5,"No classification or circuits for selectivity.",ha='center',va='center',transform=ax.transAxes)

    ax = axes[0,2]
    if circuit_activations.shape[1] >= 2:
        pca_hidden_brain = PCA(n_components=2); circuit_pca = pca_hidden_brain.fit_transform(circuit_activations)
        cmap_brain = 'viridis' if num_binary_clf_tasks == 0 or test_Y_clf_labels_subset_for_selectivity is None else 'tab10'
        scatter = ax.scatter(circuit_pca[:,0], circuit_pca[:,1], c=test_Y_clf_labels_subset_for_selectivity, cmap=cmap_brain, alpha=0.6, s=15)
        if num_binary_clf_tasks > 0 and test_Y_clf_labels_subset_for_selectivity is not None: plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Task C1 Label")
        ax.set_xlabel(f'PC1 ({pca_hidden_brain.explained_variance_ratio_[0]:.1%} var)'); ax.set_ylabel(f'PC2 ({pca_hidden_brain.explained_variance_ratio_[1]:.1%} var)'); ax.set_title('Population Vector (PCA of Outputs)')
    else: ax.text(0.5,0.5,"Need >= 2 circuits for PCA.",ha='center',va='center',transform=ax.transAxes)

    ax = axes[1,0]; epochs_plot = learner.training_history['epoch']; activation_evolution = np.array(learner.training_history['circuit_activations_mean_epoch'])
    if epochs_plot and activation_evolution.ndim == 2 and activation_evolution.shape[1] > 0:
        n_circuits_to_show = min(5, activation_evolution.shape[1]); circuit_indices = np.linspace(0, activation_evolution.shape[1]-1, n_circuits_to_show, dtype=int)
        for i, circuit_idx in enumerate(circuit_indices): ax.plot(epochs_plot, activation_evolution[:,circuit_idx], label=f'Circuit {circuit_idx}', alpha=0.8, linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Output'); ax.set_title('Circuit Output Evolution'); ax.legend(fontsize='small'); ax.grid(True,alpha=0.3)
    else: ax.text(0.5,0.5,"No activation history.",ha='center',va='center',transform=ax.transAxes)

    ax = axes[1,1]
    if circuit_activations.shape[1] > 1:
        circuit_corr = np.corrcoef(circuit_activations.T); mask = np.triu(np.ones_like(circuit_corr, dtype=bool), k=1); correlations_brain = circuit_corr[mask]
        if correlations_brain.size > 0 and not np.all(np.isnan(correlations_brain)):
            mean_corr = np.nanmean(correlations_brain)
            sns.histplot(correlations_brain[~np.isnan(correlations_brain)], bins=20, kde=True, ax=ax, color='lightgreen');
            ax.axvline(mean_corr, color='red', linestyle='--', label=f'Mean: {mean_corr:.2f}');
            ax.set_xlabel('Pairwise Correlation (Outputs)'); ax.set_ylabel('Density'); ax.set_title('Circuit Output Redundancy'); ax.legend(fontsize='small'); ax.grid(True,alpha=0.3)
        else: ax.text(0.5,0.5,"Not enough data/all NaNs for correlations.",ha='center',va='center',transform=ax.transAxes)
    else: ax.text(0.5,0.5,"Need >1 circuit for redundancy.",ha='center',va='center',transform=ax.transAxes)

    ax = axes[1,2]
    if circuit_activations.shape[1] > 0: info_efficiency_proxy = np.var(circuit_activations, axis=0); ax.bar(range(len(info_efficiency_proxy)), info_efficiency_proxy, alpha=0.7, color='gold'); ax.set_xlabel('Circuit Index'); ax.set_ylabel('Output Variance (Proxy)'); ax.set_title('Circuit "Information" (Output Var)'); ax.grid(True,alpha=0.3)
    else: ax.text(0.5,0.5,"No circuits for info efficiency.",ha='center',va='center',transform=ax.transAxes)

    plt.suptitle('Brain-like Properties Analysis (Test Data Subset - Circuit Outputs)', fontsize=16, y=0.98); plt.tight_layout(rect=[0,0,1,0.95]); plt.show()
    print("\n🧠 Brain-like Properties Summary (Test Data Subset - Circuit Outputs):")
    if 'sparsity_levels' in locals() and sparsity_levels: print(f"Average Output Sparsity (<{threshold_sparsity} act.): {np.mean(sparsity_levels):.3f}")
    if 'selectivity_scores' in locals() and selectivity_scores: print(f"Average Output Selectivity (CV for Task C1): {np.mean(selectivity_scores):.3f}")
    if 'correlations_brain' in locals() and correlations_brain.size > 0 and not np.all(np.isnan(correlations_brain)): print(f"Average Circuit Output Correlation: {np.nanmean(correlations_brain):.3f}")
    if 'pca_hidden_brain' in locals() and hasattr(pca_hidden_brain, 'explained_variance_ratio_') and pca_hidden_brain.explained_variance_ratio_.size >=2 : print(f"Population Output Variance Explained (PC1+PC2): {np.sum(pca_hidden_brain.explained_variance_ratio_[:2]):.1%}")

def plot_learning_dynamics(learner):
    if not learner.training_history['epoch']: print("Skipping learning dynamics plot: No training history."); return
    fig, axes = plt.subplots(2,2,figsize=(15,10)); fig.subplots_adjust(hspace=0.35,wspace=0.25); epochs_plot = learner.training_history['epoch']
    ax = axes[0,0]
    if learner.n_outputs_regression > 0: ax.plot(epochs_plot, learner.training_history['loss_reg'], label='Reg Loss (Train)', linewidth=1.5)
    if learner.n_outputs_classification > 0: ax.plot(epochs_plot, learner.training_history['loss_clf'], label='Clf Loss (Train)', linewidth=1.5)
    if learner.l1_activation_lambda > 0 and 'loss_l1_act' in learner.training_history and learner.training_history['loss_l1_act']: ax.plot(epochs_plot, learner.training_history['loss_l1_act'], label='L1 Act Loss (Train)', linewidth=1.5, linestyle=':')
    ax.plot(epochs_plot, learner.training_history['loss'], label='Total Loss (Train)', linewidth=2, linestyle='-')
    if learner.training_history['val_loss'] and len(learner.training_history['val_loss'])==len(epochs_plot):
        if learner.n_outputs_regression > 0: ax.plot(epochs_plot, learner.training_history['val_loss_reg'], label='Reg Loss (Val)', linestyle='--', linewidth=1.5)
        if learner.n_outputs_classification > 0: ax.plot(epochs_plot, learner.training_history['val_loss_clf'], label='Clf Loss (Val)', linestyle='--', linewidth=1.5)
        ax.plot(epochs_plot, learner.training_history['val_loss'], label='Total Loss (Val)', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss Component Evolution'); ax.legend(fontsize='small'); ax.grid(True,alpha=0.3);
    # Consider yscale log if appropriate
    # if any(l > 0 for l in learner.training_history['loss']): ax.set_yscale('log', nonpositive='clip')


    ax = axes[0,1]
    if learner.training_history['gradient_norms_w1']: ax.plot(epochs_plot, learner.training_history['gradient_norms_w1'], label='Grad W1', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_regression > 0 and learner.training_history['gradient_norms_w2_reg']: ax.plot(epochs_plot, learner.training_history['gradient_norms_w2_reg'], label='Grad W2_Reg', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_classification > 0 and learner.training_history['gradient_norms_w2_clf']: ax.plot(epochs_plot, learner.training_history['gradient_norms_w2_clf'], label='Grad W2_Clf', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient Norm (Log Scale)'); ax.set_title('Global Gradient Magnitude'); ax.legend(fontsize='small'); ax.grid(True,alpha=0.3)
    if any(g > 1e-9 for g in learner.training_history.get('gradient_norms_w1', [0])) or \
       (learner.n_outputs_regression > 0 and any(g > 1e-9 for g in learner.training_history.get('gradient_norms_w2_reg', [0]))) or \
       (learner.n_outputs_classification > 0 and any(g > 1e-9 for g in learner.training_history.get('gradient_norms_w2_clf', [0]))):
        ax.set_yscale('log',nonpositive='clip')

    ax = axes[1,0]
    if learner.training_history['weight_changes_w1']: ax.plot(epochs_plot, learner.training_history['weight_changes_w1'], label='ΔW1', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_regression > 0 and learner.training_history['weight_changes_w2_reg']: ax.plot(epochs_plot, learner.training_history['weight_changes_w2_reg'], label='ΔW2_Reg', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_classification > 0 and learner.training_history['weight_changes_w2_clf']: ax.plot(epochs_plot, learner.training_history['weight_changes_w2_clf'], label='ΔW2_Clf', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Weight Change Mag (Log Scale)'); ax.set_title('Global Weight Update Dynamics'); ax.legend(fontsize='small'); ax.grid(True,alpha=0.3)
    if any(c > 1e-9 for c in learner.training_history.get('weight_changes_w1', [0])) or \
       (learner.n_outputs_regression > 0 and any(c > 1e-9 for c in learner.training_history.get('weight_changes_w2_reg', [0]))) or \
       (learner.n_outputs_classification > 0 and any(c > 1e-9 for c in learner.training_history.get('weight_changes_w2_clf', [0]))):
        ax.set_yscale('log',nonpositive='clip')

    ax = axes[1,1]
    if len(epochs_plot) > 1:
        loss_changes = -np.diff(learner.training_history['loss']); total_weight_change_epoch = np.zeros_like(loss_changes,dtype=float)
        if learner.training_history['weight_changes_w1'] and len(learner.training_history['weight_changes_w1']) > 1: total_weight_change_epoch += np.array(learner.training_history['weight_changes_w1'][1:])
        if learner.n_outputs_regression > 0 and learner.training_history['weight_changes_w2_reg'] and len(learner.training_history['weight_changes_w2_reg']) > 1: total_weight_change_epoch += np.array(learner.training_history['weight_changes_w2_reg'][1:])
        if learner.n_outputs_classification > 0 and learner.training_history['weight_changes_w2_clf'] and len(learner.training_history['weight_changes_w2_clf']) > 1: total_weight_change_epoch += np.array(learner.training_history['weight_changes_w2_clf'][1:])
        if total_weight_change_epoch.size == loss_changes.size and np.any(total_weight_change_epoch > 1e-9): # Avoid division by zero
            efficiency = loss_changes / (total_weight_change_epoch + 1e-9)
            window_size = min(len(efficiency), max(1, len(epochs_plot)//20))
            if window_size > 1 and len(efficiency) >= window_size: efficiency_smoothed = np.convolve(efficiency, np.ones(window_size)/window_size, mode='valid'); ax.plot(epochs_plot[1:len(efficiency_smoothed)+1], efficiency_smoothed, color='darkcyan', linewidth=2)
            elif len(efficiency) > 0 : ax.plot(epochs_plot[1:len(efficiency)+1], efficiency, color='darkcyan', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Efficiency (ΔLoss / ΔWeight)'); ax.set_title('Learning Efficiency (Smoothed)'); ax.grid(True,alpha=0.3); ax.axhline(0,color='grey',linestyle=':',linewidth=1)
    plt.suptitle('Learning Dynamics Analysis', fontsize=16, y=0.98); plt.tight_layout(rect=[0,0,1,0.95]); plt.show()


# --- Many Simple Tasks Data Generation (same) ---
def generate_many_simple_tasks_data(
    n_samples: int = 1000, seed: int = 42,
    n_reg_tasks: int = 3, n_binary_clf_tasks: int = 2,
    noise_level_reg: float = 0.05, noise_level_clf_flip_prob: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x0 = np.random.uniform(-2, 2, n_samples); x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(0, 1, n_samples); x3 = np.sin(np.pi * x0 * 0.8) + 0.5 * np.cos(np.pi * x1 * 0.6)
    x4 = x0 * x1 * np.exp(-0.1 * x2**2); x5 = np.random.normal(0, 0.3, n_samples)
    inputs = np.stack([x0, x1, x2, x3, x4, x5], axis=1)
    targets_reg = np.zeros((n_samples, n_reg_tasks))
    if n_reg_tasks >= 1: targets_reg[:, 0] = (x0 + x1) / 2 + x2 * 0.5 + np.random.normal(0, noise_level_reg, n_samples)
    if n_reg_tasks >= 2: targets_reg[:, 1] = np.tanh(x2 * 2 - 1) + x3 * 0.3 + np.random.normal(0, noise_level_reg, n_samples)
    if n_reg_tasks >= 3: targets_reg[:, 2] = np.cos(x3 * np.pi) * (x4 / (np.std(x4) + 1e-6)) * 0.5 + np.random.normal(0, noise_level_reg, n_samples)
    for i in range(n_reg_tasks): targets_reg[:, i] = (targets_reg[:, i] - np.mean(targets_reg[:, i])) / (np.std(targets_reg[:, i]) + 1e-6)
    targets_clf_binary_labels = np.zeros((n_samples, n_binary_clf_tasks), dtype=int)
    if n_binary_clf_tasks >= 1: targets_clf_binary_labels[:, 0] = (x0 > 0).astype(int)
    if n_binary_clf_tasks >= 2: targets_clf_binary_labels[:, 1] = (x2 > 0.6).astype(int)
    if n_binary_clf_tasks >= 3: targets_clf_binary_labels[:, 2] = (x3 > 0).astype(int)
    for i in range(n_binary_clf_tasks):
        flip_indices = np.random.rand(n_samples) < noise_level_clf_flip_prob
        targets_clf_binary_labels[flip_indices, i] = 1 - targets_clf_binary_labels[flip_indices, i]
    return inputs, targets_reg, targets_clf_binary_labels

# --- Metrics (same) ---
def r_squared(y_true, y_pred):
    if y_true.size == 0 or y_pred.size == 0: return 0.0
    ss_res = np.sum((y_true - y_pred)**2)
    mean_y_true = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mean_y_true)**2)
    if ss_tot == 0: return 1.0 if ss_res < 1e-9 else 0.0
    return 1.0 - (ss_res / ss_tot)

def accuracy_score_multi_binary(y_true_binary_labels, y_pred_sigmoid_probs, num_binary_tasks):
    if y_true_binary_labels.size == 0 or y_pred_sigmoid_probs.size == 0 or num_binary_tasks == 0: return 0.0
    accuracies = []
    for i in range(num_binary_tasks):
        if y_true_binary_labels.ndim == 2 and y_true_binary_labels.shape[1] > i and \
           y_pred_sigmoid_probs.ndim == 2 and y_pred_sigmoid_probs.shape[1] > i:
            true_labels_task_i = y_true_binary_labels[:, i].flatten()
            pred_labels_task_i = (y_pred_sigmoid_probs[:, i] > 0.5).astype(int)
            accuracies.append(np.mean(true_labels_task_i == pred_labels_task_i))
    return np.mean(accuracies) if accuracies else 0.0


# --- Main Execution ---
if __name__ == "__main__":
    print("🧠 Multi-Task Complex Learner - ENHANCED with Local MC Learning Demo")
    print("=" * 70)

    NUM_SAMPLES = 2000 # Reduced for quicker demo of new features
    TRAIN_SPLIT_RATIO = 0.8; VALIDATION_SPLIT_TRAIN = 0.20
    N_REG_TASKS = 2; N_BINARY_CLF_TASKS = 1
    N_HIDDEN_CIRCUITS = 20; N_INTERNAL_UNITS = 4
    LEARNING_RATE = 0.0005 # Global LR
    N_EPOCHS = 1000; BATCH_SIZE = 32; CLIP_GRAD_VAL = 1.0 # Reduced epochs
    MIN_EPOCHS_NO_IMPROVE = 75; PATIENCE_NO_IMPROVE = 150 # Adjusted patience
    RANDOM_SEED_DATA = 2029; RANDOM_SEED_NETWORK = 106

    # New parameters for MicroCircuit internals and Learner
    INTERNAL_ACTIVATION = "leaky_relu" # "relu" or "leaky_relu"
    LEAKY_RELU_ALPHA = 0.05
    LR_MC_WEIGHTS = 1e-8  # Learning rate for internal MC weights (Hebbian)
    LR_MC_BIASES = 1e-8   # Learning rate for internal MC biases
    MC_WEIGHT_DECAY = 0.1 # Decay for MC internal weights to stabilize Hebbian

    MICROCIRCUIT_AGGREGATION = "max"; L1_ACTIVATION_LAMBDA = 0.00000 # Start with no L1
    USE_KWTA_ON_CIRCUITS = True; KWTA_K_CIRCUITS = 8
    LOSS_WEIGHT_REGRESSION = 1.0; LOSS_WEIGHT_CLASSIFICATION = 1.0

    N_FEATURES_MANY_TASKS = 6
    N_OUTPUTS_REGRESSION = N_REG_TASKS

    inputs, targets_reg, targets_clf_binary_labels = generate_many_simple_tasks_data(
        NUM_SAMPLES, seed=RANDOM_SEED_DATA, n_reg_tasks=N_REG_TASKS, n_binary_clf_tasks=N_BINARY_CLF_TASKS,
        noise_level_reg=0.05, noise_level_clf_flip_prob=0.02)

    n_train_val = int(NUM_SAMPLES * TRAIN_SPLIT_RATIO)
    train_val_X, train_val_Y_reg, train_val_Y_clf_binary_labels = inputs[:n_train_val], targets_reg[:n_train_val], targets_clf_binary_labels[:n_train_val]
    test_X, test_Y_reg, test_Y_clf_binary_labels = inputs[n_train_val:], targets_reg[n_train_val:], targets_clf_binary_labels[n_train_val:]

    print(f"Generated {NUM_SAMPLES} samples with {inputs.shape[1]} input features.")
    print(f"Targeting {N_REG_TASKS} regression tasks and {N_BINARY_CLF_TASKS} binary classification tasks.")
    n_inputs_actual = train_val_X.shape[1]

    learner = MultiTaskComplexLearner(
        n_inputs=n_inputs_actual, n_outputs_regression=N_OUTPUTS_REGRESSION,
        n_binary_clf_tasks=N_BINARY_CLF_TASKS,
        n_hidden_circuits=N_HIDDEN_CIRCUITS, n_internal_units_per_circuit=N_INTERNAL_UNITS,
        learning_rate=LEARNING_RATE,
        loss_weight_regression=LOSS_WEIGHT_REGRESSION,
        loss_weight_classification=LOSS_WEIGHT_CLASSIFICATION,
        microcircuit_aggregation=MICROCIRCUIT_AGGREGATION,
        internal_activation_type=INTERNAL_ACTIVATION,
        leaky_relu_alpha=LEAKY_RELU_ALPHA,
        lr_internal_weights=LR_MC_WEIGHTS,
        lr_internal_biases=LR_MC_BIASES,
        internal_weight_decay=MC_WEIGHT_DECAY,
        l1_activation_lambda=L1_ACTIVATION_LAMBDA,
        use_kwta_on_circuits=USE_KWTA_ON_CIRCUITS, kwta_k_circuits=KWTA_K_CIRCUITS,
        lr_scheduler_patience=50, lr_scheduler_factor=0.5, min_lr=1e-7, # LR scheduler params
        seed=RANDOM_SEED_NETWORK)

    print(f"\nInitialized Learner: {learner.n_hidden_circuits} circuits, {learner.n_internal_units_per_circuit} internal units.")
    print(f"MC Internal Activation: {INTERNAL_ACTIVATION}, MC Aggregation: {MICROCIRCUIT_AGGREGATION}")
    print(f"MC Internal LR (W/B): {LR_MC_WEIGHTS}/{LR_MC_BIASES}, MC Weight Decay: {MC_WEIGHT_DECAY}")
    print(f"Global LR: {LEARNING_RATE}, L1 Lambda: {L1_ACTIVATION_LAMBDA}, kWTA: {USE_KWTA_ON_CIRCUITS} (k={KWTA_K_CIRCUITS})")


    print("\n🚀 Starting Training...")
    import time; start_time = time.time()
    learner.learn(
        train_val_X, train_val_Y_reg, train_val_Y_clf_binary_labels,
        n_epochs=N_EPOCHS, min_epochs_no_improve=MIN_EPOCHS_NO_IMPROVE,
        patience_no_improve=PATIENCE_NO_IMPROVE, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT_TRAIN, verbose=True, clip_val=CLIP_GRAD_VAL)
    end_time = time.time(); print(f"--- Training finished in {end_time - start_time:.2f} seconds ---")

    title_prefix_plots = f"EnhancedMC(Hebbian) Agg={MICROCIRCUIT_AGGREGATION}, MCAct={INTERNAL_ACTIVATION}, L1={L1_ACTIVATION_LAMBDA}, kWTA={USE_KWTA_ON_CIRCUITS}(k={KWTA_K_CIRCUITS if USE_KWTA_ON_CIRCUITS else 'N/A'}) LR={LEARNING_RATE} "
    plot_comprehensive_analysis(learner, test_X, test_Y_reg, test_Y_clf_binary_labels, N_BINARY_CLF_TASKS, title_prefix=title_prefix_plots)
    plot_learning_dynamics(learner)
    if test_X.shape[0] > 0: plot_brain_like_analysis(learner, test_X, test_Y_reg, test_Y_clf_binary_labels, N_BINARY_CLF_TASKS)

    print("\n🧪 Evaluating on Test Data (Console Summary)...")
    if test_X.shape[0] > 0:
        test_preds_reg, test_preds_clf_sigmoid_probs, _ = learner.predict(test_X, batch_size=BATCH_SIZE)
        print("\n--- Regression Tasks ---")
        for i in range(N_REG_TASKS):
            if test_Y_reg.ndim == 2 and test_Y_reg.shape[1] > i and test_preds_reg.shape[1] > i:
                mse_task = np.mean((test_Y_reg[:, i] - test_preds_reg[:, i]) ** 2)
                r2_task = r_squared(test_Y_reg[:, i], test_preds_reg[:, i])
                print(f"Regression Task {i+1}: MSE={mse_task:.4f}, R²={r2_task:.4f}")
        print("\n--- Binary Classification Tasks ---")
        avg_clf_acc = accuracy_score_multi_binary(test_Y_clf_binary_labels, test_preds_clf_sigmoid_probs, N_BINARY_CLF_TASKS)
        print(f"Average Accuracy across {N_BINARY_CLF_TASKS} binary tasks: {avg_clf_acc:.4f}")
        for i in range(N_BINARY_CLF_TASKS):
             if test_Y_clf_binary_labels.ndim == 2 and test_Y_clf_binary_labels.shape[1] > i and \
                test_preds_clf_sigmoid_probs.ndim == 2 and test_preds_clf_sigmoid_probs.shape[1] > i:
                true_labels_task_i = test_Y_clf_binary_labels[:, i].flatten()
                pred_labels_task_i = (test_preds_clf_sigmoid_probs[:, i] > 0.5).astype(int)
                acc_task_i = np.mean(true_labels_task_i == pred_labels_task_i)
                print(f"Binary Classification Task {i+1} Accuracy: {acc_task_i:.4f}")
    else: print("No test data to evaluate for console summary.")

    print("\n--- MicroCircuit Internal Parameter Summary (Final State) ---")
    mc_internal_weights_norms = learner.get_microcircuit_internal_weights_norms()
    mc_internal_biases_means = learner.get_microcircuit_internal_biases_means()
    if mc_internal_weights_norms:
        print(f"Mean L2 norm of MC internal weights: {np.mean(mc_internal_weights_norms):.4f} (Std: {np.std(mc_internal_weights_norms):.4f})")
    if mc_internal_biases_means:
        print(f"Mean of MC internal biases: {np.mean(mc_internal_biases_means):.4f} (Std: {np.std(mc_internal_biases_means):.4f})")

    print("\n✅ Enhanced Multi-Task Demo Completed.")
