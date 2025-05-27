import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Any
from numba import jit
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE # Not strictly used
import warnings
warnings.filterwarnings('ignore')

# --- Utility Functions (ReLU, Softmax, CCE, kWTA - remain same) ---
@jit(nopython=True, cache=True)
def relu_numba(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

@jit(nopython=True, cache=True)
def relu_derivative_numba(x_activated: np.ndarray) -> np.ndarray: 
    return (x_activated > 0.0).astype(x_activated.dtype)

@jit(nopython=True, cache=True)
def softmax_numba(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        if x.shape[0] > 0:
            x_max_scalar = np.max(x)
        else:
            x_max_scalar = 0.0
        e_x = np.exp(x - x_max_scalar)
        sum_e_x_scalar = np.sum(e_x)
        if sum_e_x_scalar == 0:
            return np.full_like(x, 1.0 / x.shape[0]) if x.shape[0] > 0 else x
        return e_x / sum_e_x_scalar
    x_max_flat = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        if x.shape[1] > 0:
            x_max_flat[i] = np.max(x[i, :])
        else:
            x_max_flat[i] = 0.0
    x_max_reshaped = x_max_flat.reshape(x.shape[0], 1)
    e_x = np.exp(x - x_max_reshaped)
    sum_e_x_flat = np.empty(x.shape[0], dtype=e_x.dtype)
    for i in range(x.shape[0]):
        sum_e_x_flat[i] = np.sum(e_x[i, :])
    sum_e_x_reshaped = sum_e_x_flat.reshape(x.shape[0], 1)
    result = np.zeros_like(e_x)
    for i in range(x.shape[0]):
        if sum_e_x_reshaped[i, 0] != 0:
            result[i, :] = e_x[i, :] / sum_e_x_reshaped[i, 0]
        else:
            if x.shape[1] > 0:
                 result[i, :] = 1.0 / x.shape[1]
    return result

@jit(nopython=True, cache=True)
def categorical_cross_entropy_numba(
    y_true_one_hot: np.ndarray, y_pred_logits: np.ndarray
) -> float:
    if y_pred_logits.ndim == 1:
        if y_pred_logits.shape[0] == 0: return 0.0
        max_logit = np.max(y_pred_logits)
        exp_logits_shifted = np.exp(y_pred_logits - max_logit)
        log_sum_exp = max_logit + np.log(np.sum(exp_logits_shifted) + 1e-9)
        log_likelihood = np.dot(y_true_one_hot, y_pred_logits) - log_sum_exp
        return -log_likelihood
    else:
        batch_size = y_pred_logits.shape[0]
        if batch_size == 0: return 0.0
        max_logits_flat = np.empty(batch_size, dtype=y_pred_logits.dtype)
        for i in range(batch_size):
            if y_pred_logits.shape[1] > 0:
                max_logits_flat[i] = np.max(y_pred_logits[i, :])
            else:
                max_logits_flat[i] = 0.0
        max_logits_reshaped = max_logits_flat.reshape(batch_size, 1)
        exp_terms_shifted = np.exp(y_pred_logits - max_logits_reshaped)
        sum_exp_terms_flat = np.empty(batch_size, dtype=exp_terms_shifted.dtype)
        for i in range(batch_size):
            sum_exp_terms_flat[i] = np.sum(exp_terms_shifted[i, :])
        epsilon = 1e-9
        log_sum_exp_reshaped = max_logits_reshaped + \
                               np.log(sum_exp_terms_flat.reshape(batch_size, 1) + epsilon)
        dot_product_per_row_flat = np.empty(batch_size, dtype=y_pred_logits.dtype)
        for i in range(batch_size):
            dot_product_per_row_flat[i] = np.dot(y_true_one_hot[i,:], y_pred_logits[i,:])
        dot_product_per_row_reshaped = dot_product_per_row_flat.reshape(batch_size, 1)
        log_likelihoods_batch = dot_product_per_row_reshaped - log_sum_exp_reshaped
        return -np.sum(log_likelihoods_batch) / batch_size

@jit(nopython=True, cache=True)
def kwta_numba(activations: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= activations.shape[0]: 
        if k <= 0: return np.zeros_like(activations)
        return activations.copy() 
    result = np.zeros_like(activations)
    indices = np.argsort(activations)[-k:] 
    for i in range(indices.shape[0]): 
        result[indices[i]] = activations[indices[i]]
    return result

# --- MicroCircuit Class (remains same) ---
class MicroCircuit:
    def __init__(self, n_internal_units: int = 3, input_scale: float = 1.0, aggregation: str = "mean"):
        self.n_internal_units = n_internal_units
        self.internal_weights = np.random.randn(n_internal_units).astype(np.float64) * input_scale
        self.internal_biases = np.random.randn(n_internal_units).astype(np.float64) * 0.1 
        self.aggregation = aggregation
        if aggregation not in ["mean", "max"]:
            raise ValueError("Aggregation must be 'mean' or 'max'")
        
    def activate(self, circuit_input_scalar: float) -> Tuple[float, np.ndarray, np.ndarray]:
        circuit_input_scalar_f64 = np.float64(circuit_input_scalar)
        internal_pre_activations = self.internal_weights * circuit_input_scalar_f64 + self.internal_biases
        internal_activations = relu_numba(internal_pre_activations) 
        
        circuit_output = 0.0
        if internal_activations.size > 0:
            if self.aggregation == "mean":
                circuit_output = np.mean(internal_activations)
            elif self.aggregation == "max":
                circuit_output = np.max(internal_activations)
        return circuit_output, internal_pre_activations, internal_activations

# --- Static function for backward pass (remains same) ---
@jit(nopython=True, cache=True)
def _backward_pass_static_jitted(
        pred_reg_np, target_reg_np,
        pred_clf_probs_np, target_clf_one_hot_np, 
        x_np, a_h_shared_np, 
        all_internal_activations_sample_np, 
        W1_np, W2_reg_np, W2_clf_np, 
        hidden_circuits_iw_np, 
        n_hidden_circuits, n_internal_units_per_circuit,
        loss_weight_reg, loss_weight_clf,
        microcircuit_aggregation_method_code: int, 
        l1_activation_lambda: float
    ):
    error_output_layer_reg = pred_reg_np - target_reg_np 
    dW2_reg = np.outer(a_h_shared_np, error_output_layer_reg) 
    db2_reg = error_output_layer_reg 
    error_propagated_to_hidden_reg = np.dot(error_output_layer_reg, W2_reg_np.T) * loss_weight_reg
    error_output_layer_clf = pred_clf_probs_np - target_clf_one_hot_np 
    dW2_clf = np.outer(a_h_shared_np, error_output_layer_clf) 
    db2_clf = error_output_layer_clf 
    error_propagated_to_hidden_clf = np.dot(error_output_layer_clf, W2_clf_np.T) * loss_weight_clf
    total_error_propagated_to_hidden_outputs = error_propagated_to_hidden_reg + error_propagated_to_hidden_clf
    if l1_activation_lambda > 0:
        grad_l1_penalty = l1_activation_lambda * np.sign(a_h_shared_np)
        total_error_propagated_to_hidden_outputs += grad_l1_penalty
    dL_dz_h_shared = np.zeros(n_hidden_circuits, dtype=np.float64)
    for i in range(n_hidden_circuits):
        internal_activations_i = all_internal_activations_sample_np[i, :] 
        if internal_activations_i.size > 0:
            ds_dz_internal = relu_derivative_numba(internal_activations_i) 
            circuit_derivative = 0.0
            if microcircuit_aggregation_method_code == 0: # MEAN
                circuit_derivative = np.mean(ds_dz_internal * hidden_circuits_iw_np[i, :])
            elif microcircuit_aggregation_method_code == 1: # MAX
                idx_max = np.argmax(internal_activations_i) 
                circuit_derivative = ds_dz_internal[idx_max] * hidden_circuits_iw_np[i, idx_max]
            dL_dz_h_shared[i] = total_error_propagated_to_hidden_outputs[i] * circuit_derivative
        else:
            dL_dz_h_shared[i] = 0.0
    dW1 = np.outer(x_np, dL_dz_h_shared) 
    db1 = dL_dz_h_shared 
    return dW1, db1, dW2_reg, db2_reg, dW2_clf, db2_clf

# --- MultiTaskComplexLearner Class (remains same) ---
class MultiTaskComplexLearner:
    def __init__(self, 
                 n_inputs: int, 
                 n_outputs_regression: int,
                 n_outputs_classification: int,
                 n_hidden_circuits: int = 10,
                 n_internal_units_per_circuit: int = 5,
                 learning_rate: float = 0.001,
                 loss_weight_regression: float = 1.0,
                 loss_weight_classification: float = 1.0,
                 microcircuit_aggregation: str = "mean", 
                 l1_activation_lambda: float = 0.0,   
                 use_kwta_on_circuits: bool = False,
                 kwta_k_circuits: int = 0,            
                 seed: int = 42):
        np.random.seed(seed)
        self.n_inputs = n_inputs
        self.n_outputs_regression = n_outputs_regression
        self.n_outputs_classification = n_outputs_classification
        self.n_hidden_circuits = n_hidden_circuits
        self.n_internal_units_per_circuit = n_internal_units_per_circuit
        self.learning_rate_initial = learning_rate 
        self.learning_rate = learning_rate
        self.loss_weight_regression = loss_weight_regression
        self.loss_weight_classification = loss_weight_classification
        self.microcircuit_aggregation = microcircuit_aggregation
        self.microcircuit_aggregation_code = 0 if microcircuit_aggregation == "mean" else 1
        self.l1_activation_lambda = l1_activation_lambda
        self.use_kwta_on_circuits = use_kwta_on_circuits
        self.kwta_k_circuits = kwta_k_circuits if use_kwta_on_circuits else n_hidden_circuits 
        if use_kwta_on_circuits and (kwta_k_circuits <= 0 or kwta_k_circuits > n_hidden_circuits) :
            print(f"Warning: kwta_k_circuits ({kwta_k_circuits}) invalid. Disabling kWTA.")
            self.use_kwta_on_circuits = False
            self.kwta_k_circuits = n_hidden_circuits
        self.hidden_circuits_internal_weights = np.empty((n_hidden_circuits, n_internal_units_per_circuit), dtype=np.float64)
        self.hidden_circuits_internal_biases = np.empty((n_hidden_circuits, n_internal_units_per_circuit), dtype=np.float64)
        limit_w1 = np.sqrt(6.0 / (n_inputs + n_hidden_circuits)) 
        self.W1 = np.random.uniform(-limit_w1, limit_w1, (n_inputs, n_hidden_circuits)).astype(np.float64)
        self.b1 = np.zeros(n_hidden_circuits, dtype=np.float64) 
        for i in range(n_hidden_circuits):
            mc = MicroCircuit(n_internal_units_per_circuit, input_scale=1.0, aggregation=self.microcircuit_aggregation) 
            self.hidden_circuits_internal_weights[i,:] = mc.internal_weights
            self.hidden_circuits_internal_biases[i,:] = mc.internal_biases
        if n_outputs_regression > 0:
            limit_w2_reg = np.sqrt(6.0 / (n_hidden_circuits + n_outputs_regression))
            self.W2_reg = np.random.uniform(-limit_w2_reg, limit_w2_reg, (n_hidden_circuits, n_outputs_regression)).astype(np.float64)
            self.b2_reg = np.zeros(n_outputs_regression, dtype=np.float64)
        else: 
            self.W2_reg = np.empty((n_hidden_circuits, 0), dtype=np.float64)
            self.b2_reg = np.empty(0, dtype=np.float64)
        if n_outputs_classification > 0:
            limit_w2_clf = np.sqrt(6.0 / (n_hidden_circuits + n_outputs_classification))
            self.W2_clf = np.random.uniform(-limit_w2_clf, limit_w2_clf, (n_hidden_circuits, n_outputs_classification)).astype(np.float64)
            self.b2_clf = np.zeros(n_outputs_classification, dtype=np.float64)
        else: 
            self.W2_clf = np.empty((n_hidden_circuits, 0), dtype=np.float64)
            self.b2_clf = np.empty(0, dtype=np.float64)
        self.training_history = {
            'loss': [], 'loss_reg': [], 'loss_clf':[], 'loss_l1_act': [],
            'val_loss': [], 'val_loss_reg': [], 'val_loss_clf': [],
            'epoch': [], 'learning_rate': [], 
            'gradient_norms_w1': [], 'gradient_norms_w2_reg': [], 'gradient_norms_w2_clf': [],
            'circuit_activations_mean_epoch': [], 
            'weight_changes_w1': [], 'weight_changes_w2_reg': [], 'weight_changes_w2_clf': []
        }

    @staticmethod
    @jit(nopython=True, cache=True)
    def _forward_pass_static(x_np, W1_np, b1_np, 
                             W2_reg_np, b2_reg_np, 
                             W2_clf_np, b2_clf_np,
                             hidden_circuits_iw_np, hidden_circuits_ib_np,
                             n_hidden_circuits, n_internal_units_per_circuit,
                             n_outputs_regression, n_outputs_classification,
                             microcircuit_aggregation_method_code: int, 
                             use_kwta_on_circuits: bool, kwta_k_circuits: int
                             ): 
        hidden_circuit_inputs_linear = np.dot(x_np, W1_np) + b1_np 
        a_h_shared_pre_kwta = np.zeros(n_hidden_circuits, dtype=np.float64) 
        all_internal_activations_sample = np.zeros((n_hidden_circuits, n_internal_units_per_circuit), dtype=np.float64)
        for i in range(n_hidden_circuits):
            circuit_input_scalar = np.float64(hidden_circuit_inputs_linear[i])
            internal_weights_i = hidden_circuits_iw_np[i, :]
            internal_biases_i = hidden_circuits_ib_np[i, :]
            internal_pre_activations = internal_weights_i * circuit_input_scalar + internal_biases_i
            internal_activations = relu_numba(internal_pre_activations) 
            if internal_activations.shape[0] > 0:
                if microcircuit_aggregation_method_code == 0: # MEAN
                    a_h_shared_pre_kwta[i] = np.mean(internal_activations)
                elif microcircuit_aggregation_method_code == 1: # MAX
                    a_h_shared_pre_kwta[i] = np.max(internal_activations)
            else:
                a_h_shared_pre_kwta[i] = 0.0 
            all_internal_activations_sample[i, :] = internal_activations
        a_h_shared = a_h_shared_pre_kwta 
        if use_kwta_on_circuits:
            a_h_shared = kwta_numba(a_h_shared_pre_kwta, kwta_k_circuits)
        prediction_reg = np.empty(n_outputs_regression, dtype=np.float64)
        if n_outputs_regression > 0:
            final_output_linear_reg = np.dot(a_h_shared, W2_reg_np) + b2_reg_np
            prediction_reg = final_output_linear_reg 
        logits_clf = np.empty(n_outputs_classification, dtype=np.float64)
        prediction_clf_probs = np.empty(n_outputs_classification, dtype=np.float64)
        if n_outputs_classification > 0:
            logits_clf = np.dot(a_h_shared, W2_clf_np) + b2_clf_np
            prediction_clf_probs = softmax_numba(logits_clf)
        return prediction_reg, prediction_clf_probs, logits_clf, a_h_shared, all_internal_activations_sample

    def forward_pass(self, input_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        x_np = input_params.astype(np.float64)
        pred_reg, pred_clf_probs, logits_clf, a_h_shared, all_internal_acts_sample = self._forward_pass_static(
            x_np, self.W1, self.b1, 
            self.W2_reg, self.b2_reg,
            self.W2_clf, self.b2_clf,
            self.hidden_circuits_internal_weights, self.hidden_circuits_internal_biases,
            self.n_hidden_circuits, self.n_internal_units_per_circuit,
            self.n_outputs_regression, self.n_outputs_classification,
            self.microcircuit_aggregation_code,
            self.use_kwta_on_circuits, self.kwta_k_circuits
        )
        cache = {
            'x': x_np,
            'hidden_circuit_outputs_shared': a_h_shared, 
            'all_internal_activations_sample': all_internal_acts_sample, 
            'prediction_reg': pred_reg,
            'prediction_clf_probs': pred_clf_probs,
            'logits_clf': logits_clf,
        }
        return pred_reg, pred_clf_probs, cache

    def backward_pass_to_get_grads(self, 
                                   target_reg: np.ndarray, 
                                   target_clf_one_hot: np.ndarray, 
                                   cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_np = cache['x']
        a_h_shared_np = cache['hidden_circuit_outputs_shared'] 
        all_internal_acts_sample_np = cache['all_internal_activations_sample'] 
        pred_reg_np = cache['prediction_reg']
        pred_clf_probs_np = cache['prediction_clf_probs']
        target_reg_np = target_reg.astype(np.float64)
        target_clf_one_hot_np = target_clf_one_hot.astype(np.float64)
        eff_pred_reg_np = pred_reg_np if self.n_outputs_regression > 0 else np.empty(0, dtype=np.float64)
        eff_target_reg_np = target_reg_np if self.n_outputs_regression > 0 else np.empty(0, dtype=np.float64)
        eff_pred_clf_probs_np = pred_clf_probs_np if self.n_outputs_classification > 0 else np.empty(0, dtype=np.float64)
        eff_target_clf_one_hot_np = target_clf_one_hot_np if self.n_outputs_classification > 0 else np.empty(0, dtype=np.float64)
        dW1, db1, dW2_reg, db2_reg, dW2_clf, db2_clf = _backward_pass_static_jitted(
            eff_pred_reg_np, eff_target_reg_np,
            eff_pred_clf_probs_np, eff_target_clf_one_hot_np,
            x_np, a_h_shared_np, 
            all_internal_acts_sample_np,
            self.W1, self.W2_reg, self.W2_clf, 
            self.hidden_circuits_internal_weights,
            self.n_hidden_circuits, self.n_internal_units_per_circuit,
            self.loss_weight_regression if self.n_outputs_regression > 0 else 0.0, 
            self.loss_weight_classification if self.n_outputs_classification > 0 else 0.0,
            self.microcircuit_aggregation_code,
            self.l1_activation_lambda
        )
        return dW1, db1, dW2_reg, db2_reg, dW2_clf, db2_clf

    def learn( 
        self, 
        input_data: np.ndarray, 
        target_data_reg: np.ndarray, 
        target_data_clf: np.ndarray, 
        n_epochs: int = 1000,
        min_epochs_no_improve: int = 50, 
        patience_no_improve: int = 100,  
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: bool = True,
        clip_val: float = 1.0 
    ):
        n_samples = input_data.shape[0]
        if self.n_outputs_classification > 0:
             target_data_clf_one_hot = np.eye(self.n_outputs_classification)[target_data_clf.astype(int).flatten()]
        else: 
            target_data_clf_one_hot = np.empty((n_samples, 0), dtype=np.float64)
        if validation_split > 0 and n_samples * validation_split >= 1:
            val_size = int(n_samples * validation_split)
            permutation = np.random.permutation(n_samples)
            val_input = input_data[permutation[:val_size]]
            val_target_reg = target_data_reg[permutation[:val_size]]
            val_target_clf_one_hot = target_data_clf_one_hot[permutation[:val_size]] if self.n_outputs_classification > 0 else np.empty((val_size, 0))
            train_input = input_data[permutation[val_size:]]
            train_target_reg = target_data_reg[permutation[val_size:]]
            train_target_clf_one_hot = target_data_clf_one_hot[permutation[val_size:]] if self.n_outputs_classification > 0 else np.empty((n_samples - val_size, 0))
            n_train_samples = train_input.shape[0]
            if verbose: print(f"Training on {n_train_samples} samples, validating on {val_size} samples.")
        else:
            train_input, train_target_reg, train_target_clf_one_hot = input_data, target_data_reg, target_data_clf_one_hot
            val_input, val_target_reg, val_target_clf_one_hot = None, None, None
            n_train_samples = train_input.shape[0]
            if verbose: print(f"Training on {n_train_samples} samples (no validation split).")
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = None
        for epoch in range(n_epochs):
            epoch_train_loss_total = 0.0
            epoch_train_loss_reg = 0.0
            epoch_train_loss_clf = 0.0
            epoch_train_loss_l1_act = 0.0 
            epoch_grad_norms_w1_list = []
            epoch_grad_norms_w2_reg_list = []
            epoch_grad_norms_w2_clf_list = []
            epoch_circuit_activations_list = [] 
            prev_W1 = self.W1.copy()
            prev_W2_reg = self.W2_reg.copy() if self.n_outputs_regression > 0 else None
            prev_W2_clf = self.W2_clf.copy() if self.n_outputs_classification > 0 else None
            permutation = np.random.permutation(n_train_samples) 
            shuffled_train_input = train_input[permutation]
            shuffled_train_target_reg = train_target_reg[permutation]
            shuffled_train_target_clf_one_hot = train_target_clf_one_hot[permutation] if self.n_outputs_classification > 0 else np.empty((n_train_samples,0))
            for i in range(0, n_train_samples, batch_size):
                batch_input = shuffled_train_input[i:i+batch_size]
                batch_target_reg = shuffled_train_target_reg[i:i+batch_size]
                batch_target_clf_one_hot = shuffled_train_target_clf_one_hot[i:i+batch_size] if self.n_outputs_classification > 0 else np.empty((batch_input.shape[0],0))
                current_batch_size = batch_input.shape[0]
                batch_dW1, batch_db1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
                batch_dW2_reg = np.zeros_like(self.W2_reg) if self.n_outputs_regression > 0 else None
                batch_db2_reg = np.zeros_like(self.b2_reg) if self.n_outputs_regression > 0 else None
                batch_dW2_clf = np.zeros_like(self.W2_clf) if self.n_outputs_classification > 0 else None
                batch_db2_clf = np.zeros_like(self.b2_clf) if self.n_outputs_classification > 0 else None
                for j in range(current_batch_size): 
                    single_input = batch_input[j]
                    single_target_reg = batch_target_reg[j] if self.n_outputs_regression > 0 else np.empty(0)
                    single_target_clf_one_hot = batch_target_clf_one_hot[j] if self.n_outputs_classification > 0 else np.empty(0)
                    pred_reg, pred_clf_probs, cache = self.forward_pass(single_input) 
                    epoch_circuit_activations_list.append(cache['hidden_circuit_outputs_shared'].copy())
                    loss_reg_sample = 0.0
                    if self.n_outputs_regression > 0:
                        loss_reg_sample = np.mean((single_target_reg - pred_reg) ** 2)
                    loss_clf_sample = 0.0
                    if self.n_outputs_classification > 0 and single_target_clf_one_hot.size > 0 and cache['logits_clf'].size > 0:
                        loss_clf_sample = categorical_cross_entropy_numba(
                            single_target_clf_one_hot.reshape(1, -1), 
                            cache['logits_clf'].reshape(1, -1)    
                        )
                    l1_act_penalty_sample = 0.0
                    if self.l1_activation_lambda > 0:
                        l1_act_penalty_sample = self.l1_activation_lambda * np.sum(np.abs(cache['hidden_circuit_outputs_shared']))
                    total_loss_sample = (self.loss_weight_regression * loss_reg_sample if self.n_outputs_regression > 0 else 0.0) + \
                                        (self.loss_weight_classification * loss_clf_sample if self.n_outputs_classification > 0 else 0.0) + \
                                        l1_act_penalty_sample 
                    epoch_train_loss_total += total_loss_sample
                    epoch_train_loss_reg += loss_reg_sample
                    epoch_train_loss_clf += loss_clf_sample
                    epoch_train_loss_l1_act += l1_act_penalty_sample
                    eff_target_reg = single_target_reg if self.n_outputs_regression > 0 else np.empty((0,), dtype=np.float64)
                    eff_target_clf_one_hot = single_target_clf_one_hot if self.n_outputs_classification > 0 else np.empty((0,), dtype=np.float64)
                    dW1_s, db1_s, dW2_reg_s, db2_reg_s, dW2_clf_s, db2_clf_s = \
                        self.backward_pass_to_get_grads(eff_target_reg, eff_target_clf_one_hot, cache)
                    batch_dW1 += dW1_s; batch_db1 += db1_s
                    if self.n_outputs_regression > 0:
                        batch_dW2_reg += dW2_reg_s; batch_db2_reg += db2_reg_s
                    if self.n_outputs_classification > 0:
                        batch_dW2_clf += dW2_clf_s; batch_db2_clf += db2_clf_s
                avg_dW1 = batch_dW1 / current_batch_size
                avg_db1 = batch_db1 / current_batch_size
                epoch_grad_norms_w1_list.append(np.linalg.norm(avg_dW1))
                self.W1 -= self.learning_rate * np.clip(avg_dW1, -clip_val, clip_val)
                self.b1 -= self.learning_rate * np.clip(avg_db1, -clip_val, clip_val)
                if self.n_outputs_regression > 0 and batch_dW2_reg is not None:
                    avg_dW2_reg = batch_dW2_reg / current_batch_size
                    avg_db2_reg = batch_db2_reg / current_batch_size
                    epoch_grad_norms_w2_reg_list.append(np.linalg.norm(avg_dW2_reg))
                    self.W2_reg -= self.learning_rate * np.clip(avg_dW2_reg, -clip_val, clip_val)
                    self.b2_reg -= self.learning_rate * np.clip(avg_db2_reg, -clip_val, clip_val)
                if self.n_outputs_classification > 0 and batch_dW2_clf is not None:
                    avg_dW2_clf = batch_dW2_clf / current_batch_size
                    avg_db2_clf = batch_db2_clf / current_batch_size
                    epoch_grad_norms_w2_clf_list.append(np.linalg.norm(avg_dW2_clf))
                    self.W2_clf -= self.learning_rate * np.clip(avg_dW2_clf, -clip_val, clip_val)
                    self.b2_clf -= self.learning_rate * np.clip(avg_db2_clf, -clip_val, clip_val)
            weight_change_w1 = np.linalg.norm(self.W1 - prev_W1)
            weight_change_w2_reg = np.linalg.norm(self.W2_reg - prev_W2_reg) if self.n_outputs_regression > 0 and prev_W2_reg is not None else 0.0
            weight_change_w2_clf = np.linalg.norm(self.W2_clf - prev_W2_clf) if self.n_outputs_classification > 0 and prev_W2_clf is not None else 0.0
            avg_epoch_train_loss = epoch_train_loss_total / n_train_samples if n_train_samples > 0 else 0
            avg_epoch_train_loss_reg = epoch_train_loss_reg / n_train_samples if n_train_samples > 0 else 0
            avg_epoch_train_loss_clf = epoch_train_loss_clf / n_train_samples if n_train_samples > 0 else 0
            avg_epoch_train_loss_l1_act = epoch_train_loss_l1_act / n_train_samples if n_train_samples > 0 else 0
            self.training_history['loss'].append(avg_epoch_train_loss)
            self.training_history['loss_reg'].append(avg_epoch_train_loss_reg)
            self.training_history['loss_clf'].append(avg_epoch_train_loss_clf)
            self.training_history['loss_l1_act'].append(avg_epoch_train_loss_l1_act)
            self.training_history['epoch'].append(epoch)
            self.training_history['learning_rate'].append(self.learning_rate)
            self.training_history['gradient_norms_w1'].append(np.mean(epoch_grad_norms_w1_list) if epoch_grad_norms_w1_list else 0)
            self.training_history['gradient_norms_w2_reg'].append(np.mean(epoch_grad_norms_w2_reg_list) if epoch_grad_norms_w2_reg_list else 0)
            self.training_history['gradient_norms_w2_clf'].append(np.mean(epoch_grad_norms_w2_clf_list) if epoch_grad_norms_w2_clf_list else 0)
            self.training_history['circuit_activations_mean_epoch'].append(np.mean(epoch_circuit_activations_list, axis=0) if epoch_circuit_activations_list else np.zeros(self.n_hidden_circuits))
            self.training_history['weight_changes_w1'].append(weight_change_w1)
            self.training_history['weight_changes_w2_reg'].append(weight_change_w2_reg)
            self.training_history['weight_changes_w2_clf'].append(weight_change_w2_clf)
            current_val_loss_total = avg_epoch_train_loss 
            current_val_loss_reg = avg_epoch_train_loss_reg
            current_val_loss_clf = avg_epoch_train_loss_clf
            if val_input is not None and len(val_input) > 0:
                val_preds_reg, val_preds_clf_probs, val_logits_clf = self.predict(val_input, batch_size=batch_size)
                current_val_loss_reg = 0.0
                if self.n_outputs_regression > 0:
                    current_val_loss_reg = np.mean((val_target_reg - val_preds_reg) ** 2)
                current_val_loss_clf = 0.0
                if self.n_outputs_classification > 0 and val_target_clf_one_hot.size > 0 and val_logits_clf.size > 0 :
                    current_val_loss_clf = categorical_cross_entropy_numba(val_target_clf_one_hot, val_logits_clf)
                current_val_loss_total = (self.loss_weight_regression * current_val_loss_reg if self.n_outputs_regression > 0 else 0.0) + \
                                         (self.loss_weight_classification * current_val_loss_clf if self.n_outputs_classification > 0 else 0.0)
            self.training_history['val_loss'].append(current_val_loss_total)
            self.training_history['val_loss_reg'].append(current_val_loss_reg)
            self.training_history['val_loss_clf'].append(current_val_loss_clf)
            log_interval = max(1, n_epochs // 20) if n_epochs > 0 else 1
            if verbose and (epoch + 1) % log_interval == 0:
                log_msg = (f"Epoch {epoch + 1:4d}/{n_epochs}: Train Loss={avg_epoch_train_loss:.4f} "
                           f"(Reg: {avg_epoch_train_loss_reg:.4f}, Clf: {avg_epoch_train_loss_clf:.4f}, L1Act: {avg_epoch_train_loss_l1_act:.4f})")
                if val_input is not None and len(val_input) > 0: 
                    log_msg += (f", Val Loss={current_val_loss_total:.4f} "
                                f"(Reg: {current_val_loss_reg:.4f}, Clf: {current_val_loss_clf:.4f})")
                print(log_msg)
            loss_for_stopping = current_val_loss_total 
            if loss_for_stopping < best_val_loss: 
                best_val_loss = loss_for_stopping
                epochs_without_improvement = 0
                best_weights = {'W1': self.W1.copy(), 'b1': self.b1.copy()}
                if self.n_outputs_regression > 0:
                    best_weights['W2_reg'] = self.W2_reg.copy(); best_weights['b2_reg'] = self.b2_reg.copy()
                if self.n_outputs_classification > 0:
                    best_weights['W2_clf'] = self.W2_clf.copy(); best_weights['b2_clf'] = self.b2_clf.copy()
            else:
                epochs_without_improvement += 1
            if epoch >= min_epochs_no_improve and epochs_without_improvement >= patience_no_improve:
                if verbose: print(f"Early stopping at epoch {epoch + 1}.")
                if best_weights: 
                    self.W1, self.b1 = best_weights['W1'], best_weights['b1']
                    if self.n_outputs_regression > 0: self.W2_reg, self.b2_reg = best_weights['W2_reg'], best_weights['b2_reg']
                    if self.n_outputs_classification > 0: self.W2_clf, self.b2_clf = best_weights['W2_clf'], best_weights['b2_clf']
                break
        if verbose: print(f"Training done! Best validation loss: {best_val_loss:.6f}")

    def predict(self, input_data: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = input_data.shape[0]
        predictions_reg = np.zeros((n_samples, self.n_outputs_regression)) if self.n_outputs_regression > 0 else np.empty((n_samples, 0))
        predictions_clf_probs = np.zeros((n_samples, self.n_outputs_classification)) if self.n_outputs_classification > 0 else np.empty((n_samples, 0))
        logits_clf_all = np.zeros((n_samples, self.n_outputs_classification)) if self.n_outputs_classification > 0 else np.empty((n_samples, 0))
        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0: effective_batch_size = n_samples
        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]
                current_batch_size_pred = batch_input.shape[0]
                for j in range(current_batch_size_pred): 
                    single_input = batch_input[j]
                    pred_reg_s, pred_clf_probs_s, cache_s = self.forward_pass(single_input)
                    if self.n_outputs_regression > 0:
                        predictions_reg[i+j, :] = pred_reg_s
                    if self.n_outputs_classification > 0:
                        predictions_clf_probs[i+j, :] = pred_clf_probs_s
                        logits_clf_all[i+j, :] = cache_s['logits_clf']
        return predictions_reg, predictions_clf_probs, logits_clf_all

    def get_circuit_activations(self, input_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        n_samples = input_data.shape[0]
        activations = np.zeros((n_samples, self.n_hidden_circuits))
        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0: effective_batch_size = n_samples
        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]
                current_bs = batch_input.shape[0]
                for j in range(current_bs):
                    single_input = batch_input[j]
                    _, _, cache = self.forward_pass(single_input)
                    activations[i+j] = cache['hidden_circuit_outputs_shared']
        return activations

# --- Plotting Functions (Copied from previous response, remain same) ---
def plot_comprehensive_analysis(learner, test_X, test_Y_reg, test_Y_clf_labels, title_prefix=""):
    fig = plt.figure(figsize=(22, 28)) 
    epochs = learner.training_history['epoch']
    ax1 = plt.subplot(7, 3, 1) 
    if epochs:
        plt.plot(epochs, learner.training_history['loss'], label='Total Training Loss', linewidth=2)
        if learner.training_history['val_loss'] and len(learner.training_history['val_loss']) == len(epochs):
            plt.plot(epochs, learner.training_history['val_loss'], label='Total Validation Loss', linewidth=2, linestyle='--')
    plt.ylabel('Total Loss'); plt.title('Total Loss History'); plt.legend(); plt.grid(True, alpha=0.3)
    ax2 = plt.subplot(7, 3, 2)
    if epochs and learner.n_outputs_regression > 0:
        plt.plot(epochs, learner.training_history['loss_reg'], label='Regression Training', linewidth=2)
        if learner.training_history['val_loss_reg'] and len(learner.training_history['val_loss_reg']) == len(epochs):
            plt.plot(epochs, learner.training_history['val_loss_reg'], label='Regression Validation', linewidth=2, linestyle='--')
    plt.ylabel('Regression Loss (MSE)'); plt.title('Regression Loss History'); plt.legend(); plt.grid(True, alpha=0.3)
    ax3 = plt.subplot(7, 3, 3)
    if epochs and learner.n_outputs_classification > 0:
        plt.plot(epochs, learner.training_history['loss_clf'], label='Classification Training', linewidth=2)
        if learner.training_history['val_loss_clf'] and len(learner.training_history['val_loss_clf']) == len(epochs):
            plt.plot(epochs, learner.training_history['val_loss_clf'], label='Classification Validation', linewidth=2, linestyle='--')
    plt.ylabel('Classification Loss (CCE)'); plt.title('Classification Loss History'); plt.legend(); plt.grid(True, alpha=0.3)
    ax4 = plt.subplot(7, 3, 4)
    if epochs:
        if learner.training_history['gradient_norms_w1']: plt.plot(epochs, learner.training_history['gradient_norms_w1'], label='Grad Norm W1', linewidth=1.5, alpha=0.8)
        if learner.training_history['gradient_norms_w2_reg'] and learner.n_outputs_regression > 0: plt.plot(epochs, learner.training_history['gradient_norms_w2_reg'], label='Grad Norm W2_Reg', linewidth=1.5, alpha=0.8)
        if learner.training_history['gradient_norms_w2_clf'] and learner.n_outputs_classification > 0: plt.plot(epochs, learner.training_history['gradient_norms_w2_clf'], label='Grad Norm W2_Clf', linewidth=1.5, alpha=0.8)
    plt.ylabel('Gradient Norm'); plt.title('Gradient Norm Evolution'); plt.legend(); plt.grid(True, alpha=0.3)
    if any(learner.training_history['gradient_norms_w1']) or \
       (learner.n_outputs_regression > 0 and any(learner.training_history['gradient_norms_w2_reg'])) or \
       (learner.n_outputs_classification > 0 and any(learner.training_history['gradient_norms_w2_clf'])):
        plt.yscale('log', nonpositive='clip')
    ax5 = plt.subplot(7, 3, 5)
    if epochs:
        if learner.training_history['weight_changes_w1']: plt.plot(epochs, learner.training_history['weight_changes_w1'], label='Weight Change W1', linewidth=1.5, alpha=0.8)
        if learner.training_history['weight_changes_w2_reg'] and learner.n_outputs_regression > 0: plt.plot(epochs, learner.training_history['weight_changes_w2_reg'], label='Weight Change W2_Reg', linewidth=1.5, alpha=0.8)
        if learner.training_history['weight_changes_w2_clf'] and learner.n_outputs_classification > 0: plt.plot(epochs, learner.training_history['weight_changes_w2_clf'], label='Weight Change W2_Clf', linewidth=1.5, alpha=0.8)
    plt.ylabel('Weight Change Magnitude'); plt.title('Weight Change Evolution'); plt.legend(); plt.grid(True, alpha=0.3)
    if any(learner.training_history['weight_changes_w1']) or \
       (learner.n_outputs_regression > 0 and any(learner.training_history['weight_changes_w2_reg'])) or \
       (learner.n_outputs_classification > 0 and any(learner.training_history['weight_changes_w2_clf'])):
        plt.yscale('log', nonpositive='clip')
    ax6 = plt.subplot(7, 3, 6)
    circuit_activations_epoch_mean = np.array(learner.training_history['circuit_activations_mean_epoch'])
    if circuit_activations_epoch_mean.size > 0 and circuit_activations_epoch_mean.ndim == 2:
        im = plt.imshow(circuit_activations_epoch_mean.T, aspect='auto', cmap='viridis', interpolation='nearest'); plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        plt.ylabel('Circuit Index'); plt.xlabel('Epoch'); plt.title('Mean Circuit Activations (Epoch)')
    if test_X.shape[0] > 0:
        test_preds_reg, test_preds_clf_probs, _ = learner.predict(test_X)
        ax7 = plt.subplot(7, 3, 7)
        if learner.n_outputs_regression > 0 and test_Y_reg.size > 0 and test_preds_reg.size > 0:
            plt.scatter(test_Y_reg.flatten(), test_preds_reg.flatten(), alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
            min_val = min(np.min(test_Y_reg), np.min(test_preds_reg)); max_val = max(np.max(test_Y_reg), np.max(test_preds_reg))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
            plt.xlabel('True Values (Regression)'); plt.ylabel('Predicted Values (Regression)'); plt.title('Regression: Predicted vs True'); plt.legend()
        else: ax7.text(0.5, 0.5, "No regression task or data.", ha='center', va='center', transform=ax7.transAxes)
        plt.grid(True, alpha=0.3)
        ax8 = plt.subplot(7, 3, 8)
        if learner.n_outputs_regression > 0 and test_Y_reg.size > 0 and test_preds_reg.size > 0:
            residuals = test_Y_reg.flatten() - test_preds_reg.flatten()
            plt.scatter(test_preds_reg.flatten(), residuals, alpha=0.5, s=15, edgecolors='k', linewidths=0.5); plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values (Regression)'); plt.ylabel('Residuals'); plt.title('Residual Analysis')
        else: ax8.text(0.5, 0.5, "No regression task or data.", ha='center', va='center', transform=ax8.transAxes)
        plt.grid(True, alpha=0.3)
        ax9 = plt.subplot(7, 3, 9)
        if learner.n_outputs_regression > 0 and test_Y_reg.size > 0 and test_preds_reg.size > 0:
            residuals = test_Y_reg.flatten() - test_preds_reg.flatten()
            sns.histplot(residuals, bins=30, kde=True, ax=ax9, color='skyblue')
            plt.xlabel('Residuals'); plt.ylabel('Density'); plt.title('Residual Distribution')
        else: ax9.text(0.5, 0.5, "No regression task or data.", ha='center', va='center', transform=ax9.transAxes)
        plt.grid(True, alpha=0.3)
    if test_X.shape[0] > 0:
        sample_size_for_activation_analysis = min(500, test_X.shape[0])
        circuit_activations_test = learner.get_circuit_activations(test_X[:sample_size_for_activation_analysis])
        ax10 = plt.subplot(7, 3, 10)
        if circuit_activations_test.size > 0:
            im = plt.imshow(circuit_activations_test.T, aspect='auto', cmap='plasma', interpolation='nearest'); plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)
            plt.ylabel('Circuit Index'); plt.xlabel(f'Sample Index (first {sample_size_for_activation_analysis})'); plt.title('Circuit Activations (Test Data)')
        ax11 = plt.subplot(7, 3, 11)
        if circuit_activations_test.size > 0 and circuit_activations_test.shape[1] > 1: 
            circuit_corr = np.corrcoef(circuit_activations_test.T)
            im = plt.imshow(circuit_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest'); plt.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)
            plt.xlabel('Circuit Index'); plt.ylabel('Circuit Index'); plt.title('Circuit Correlation Matrix')
        ax12 = plt.subplot(7, 3, 12)
        if circuit_activations_test.size > 0:
            circuit_means = np.mean(circuit_activations_test, axis=0); circuit_stds = np.std(circuit_activations_test, axis=0)
            x_pos = np.arange(len(circuit_means))
            plt.bar(x_pos, circuit_means, yerr=circuit_stds, alpha=0.7, capsize=3, color='teal')
            plt.xlabel('Circuit Index'); plt.ylabel('Mean Activation ± Std'); plt.title('Circuit Activation Statistics')
            if len(x_pos) > 0: plt.xticks(x_pos[::max(1, len(x_pos)//10)])
    if test_X.shape[0] > 0:
        sample_size_for_pca = min(1000, test_X.shape[0])
        test_X_sample = test_X[:sample_size_for_pca]
        test_Y_clf_labels_sample = test_Y_clf_labels[:sample_size_for_pca]
        circuit_activations_test_sample = learner.get_circuit_activations(test_X_sample)
        ax13 = plt.subplot(7, 3, 13)
        if test_X_sample.shape[1] >= 2: # PCA for input space
            pca_input = PCA(n_components=2); X_pca = pca_input.fit_transform(test_X_sample)
            scatter_colors = test_Y_clf_labels_sample.flatten() if learner.n_outputs_classification > 0 else None
            cmap_input = 'viridis' if learner.n_outputs_classification == 0 or scatter_colors is None else 'tab10'
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scatter_colors, cmap=cmap_input, alpha=0.6, s=20)
            if learner.n_outputs_classification > 0 and scatter_colors is not None: plt.colorbar(scatter, ax=ax13, fraction=0.046, pad=0.04)
            plt.xlabel(f'PC1 ({pca_input.explained_variance_ratio_[0]:.1%} var)'); plt.ylabel(f'PC2 ({pca_input.explained_variance_ratio_[1]:.1%} var)'); plt.title('Input Space (PCA)')
        elif test_X_sample.shape[1] == 1: 
             plt.scatter(test_X_sample[:,0], np.zeros_like(test_X_sample[:,0]), c=test_Y_clf_labels_sample.flatten() if learner.n_outputs_classification > 0 else None, cmap='tab10' if learner.n_outputs_classification > 0 else 'viridis', alpha=0.6, s=20)
             plt.xlabel('Input Dimension 1'); plt.title('Input Space (1D)')
        else: ax13.text(0.5, 0.5, "Input dim < 1.", ha='center', va='center', transform=ax13.transAxes)
        ax14 = plt.subplot(7, 3, 14)
        if circuit_activations_test_sample.shape[1] >= 2: # PCA for hidden space
            pca_hidden = PCA(n_components=2); hidden_pca = pca_hidden.fit_transform(circuit_activations_test_sample)
            scatter_colors_hidden = test_Y_clf_labels_sample.flatten() if learner.n_outputs_classification > 0 else None
            cmap_hidden = 'viridis' if learner.n_outputs_classification == 0 or scatter_colors_hidden is None else 'tab10'
            scatter = plt.scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=scatter_colors_hidden, cmap=cmap_hidden, alpha=0.6, s=20)
            if learner.n_outputs_classification > 0 and scatter_colors_hidden is not None: plt.colorbar(scatter, ax=ax14, fraction=0.046, pad=0.04)
            plt.xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.1%} var)'); plt.ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.1%} var)'); plt.title('Hidden Representation (PCA)')
        elif circuit_activations_test_sample.shape[1] == 1: 
             plt.scatter(circuit_activations_test_sample[:,0], np.zeros_like(circuit_activations_test_sample[:,0]), c=test_Y_clf_labels_sample.flatten() if learner.n_outputs_classification > 0 else None, cmap='tab10' if learner.n_outputs_classification > 0 else 'viridis', alpha=0.6, s=20)
             plt.xlabel('Hidden Dimension 1'); plt.title('Hidden Representation (1D)')
        else: ax14.text(0.5, 0.5, "Hidden dim < 1.", ha='center', va='center', transform=ax14.transAxes)
        ax15 = plt.subplot(7, 3, 15) # Decision Boundary
        if test_X.shape[1] == 2 and learner.n_outputs_classification > 0: # Only for 2D input
            h = 0.05 
            x_min, x_max = test_X[:, 0].min() - 0.2, test_X[:, 0].max() + 0.2
            y_min, y_max = test_X[:, 1].min() - 0.2, test_X[:, 1].max() + 0.2
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            if mesh_points.shape[0] > 0:
                max_mesh_points = 10000 
                mesh_points_sampled = mesh_points[np.random.choice(mesh_points.shape[0], min(max_mesh_points, mesh_points.shape[0]), replace=False)] if mesh_points.shape[0] > max_mesh_points else mesh_points
                _, Z_probs, _ = learner.predict(mesh_points_sampled)
                if Z_probs.shape[1] > 0:
                    Z_pred_labels = np.argmax(Z_probs, axis=1)
                    plt.scatter(mesh_points_sampled[:, 0], mesh_points_sampled[:, 1], c=Z_pred_labels, cmap='Pastel1', alpha=0.5, s=15, marker='s')
            plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y_clf_labels.flatten(), cmap='tab10', edgecolors='k', s=25, alpha=0.8, linewidths=0.5)
            plt.xlabel('Input Dimension 1'); plt.ylabel('Input Dimension 2'); plt.title('Decision Boundaries & Test Data'); plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
        else: ax15.text(0.5, 0.5, "Decision boundary plot requires\n2D input & classification task.", ha='center', va='center', transform=ax15.transAxes)
    ax16 = plt.subplot(7, 3, 16)
    if test_X.shape[0] > 0:
        test_preds_reg, test_preds_clf_probs, _ = learner.predict(test_X) # Ensure defined
        metrics_list = []; values_list = []; colors_list = ['tomato', 'sandybrown', 'mediumseagreen', 'cornflowerblue']
        if learner.n_outputs_regression > 0 and test_Y_reg.size > 0 and test_preds_reg.size > 0:
            mse = np.mean((test_Y_reg - test_preds_reg) ** 2); mae = np.mean(np.abs(test_Y_reg - test_preds_reg)); r2 = r_squared(test_Y_reg, test_preds_reg)
            metrics_list.extend(['MSE', 'MAE', 'R² (Reg)']); values_list.extend([mse, mae, r2])
        if learner.n_outputs_classification > 0 and test_Y_clf_labels.size > 0 and test_preds_clf_probs.size > 0:
            acc = accuracy_score(test_Y_clf_labels, test_preds_clf_probs)
            metrics_list.append('Accuracy (Clf)'); values_list.append(acc)
        if metrics_list:
            bars = plt.bar(metrics_list, values_list, color=colors_list[:len(metrics_list)], alpha=0.75)
            plt.ylabel('Metric Value'); plt.title('Performance Summary (Test)'); plt.xticks(rotation=15, ha="right")
            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values_list, default=1), f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
        else: ax16.text(0.5, 0.5, "No tasks with data for metrics.", ha='center', va='center', transform=ax16.transAxes)
    ax17 = plt.subplot(7, 3, 17)
    if epochs and learner.training_history['learning_rate']:
        plt.plot(epochs, learner.training_history['learning_rate'], 'g-', linewidth=2)
        plt.ylabel('Learning Rate'); plt.xlabel('Epoch'); plt.title('Learning Rate Schedule'); plt.grid(True, alpha=0.3)
    ax18 = plt.subplot(7, 3, 18)
    if learner.n_outputs_classification > 0 and test_X.shape[0] > 0 and test_Y_clf_labels.size > 0 and test_preds_clf_probs.size > 0:
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(test_Y_clf_labels.flatten(), np.argmax(test_preds_clf_probs, axis=1))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax18, xticklabels=[f"C{i}" for i in range(learner.n_outputs_classification)], yticklabels=[f"C{i}" for i in range(learner.n_outputs_classification)])
            plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title("Confusion Matrix (Test Data)")
        except ImportError: ax18.text(0.5, 0.5, "sklearn not found for CM.", ha='center', va='center', transform=ax18.transAxes)
    else: ax18.text(0.5, 0.5, "No classification task or data.", ha='center', va='center', transform=ax18.transAxes)
    plt.suptitle(f'{title_prefix}Comprehensive Neural Network Analysis', fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97]); plt.show()

def plot_brain_like_analysis(learner, test_X, test_Y_reg, test_Y_clf_labels):
    if test_X.shape[0] == 0: print("Skipping brain-like analysis: No test data."); return
    fig, axes = plt.subplots(2, 3, figsize=(18, 11)); fig.subplots_adjust(hspace=0.4, wspace=0.3)
    sample_size_analysis = min(500, test_X.shape[0])
    test_X_subset = test_X[:sample_size_analysis]; test_Y_clf_labels_subset = test_Y_clf_labels[:sample_size_analysis]
    circuit_activations = learner.get_circuit_activations(test_X_subset)
    if circuit_activations.size == 0: print("Skipping brain-like analysis: No circuit activations obtained."); return
    ax = axes[0, 0]; sparsity_levels = []
    if circuit_activations.shape[1] > 0: 
        threshold_sparsity = 1e-5 
        for i in range(circuit_activations.shape[1]):
            sparsity = np.mean(np.abs(circuit_activations[:, i]) < threshold_sparsity); sparsity_levels.append(sparsity)
        if sparsity_levels:
            ax.bar(range(len(sparsity_levels)), sparsity_levels, alpha=0.7, color='skyblue')
            ax.set_xlabel('Circuit Index'); ax.set_ylabel(f'Sparsity Level (<{threshold_sparsity} act.)'); ax.set_title('Circuit Sparsity (Near-Zero Activations)'); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
    else: ax.text(0.5, 0.5, "No circuits for sparsity analysis.", ha='center', va='center', transform=ax.transAxes)
    ax = axes[0, 1]; selectivity_scores = []
    if learner.n_outputs_classification > 0 and circuit_activations.shape[1] > 0:
        for i in range(circuit_activations.shape[1]):
            class_responses = []
            for class_idx in range(learner.n_outputs_classification):
                mask = test_Y_clf_labels_subset.flatten() == class_idx
                if np.sum(mask) > 0: mean_response = np.mean(circuit_activations[mask, i]); class_responses.append(mean_response)
            if len(class_responses) > 1 and (np.abs(np.mean(class_responses)) > 1e-8 or np.std(class_responses) > 1e-8) : 
                selectivity = np.std(class_responses) / (np.abs(np.mean(class_responses)) + 1e-8); selectivity_scores.append(selectivity)
            else: selectivity_scores.append(0)
        if selectivity_scores:
            ax.bar(range(len(selectivity_scores)), selectivity_scores, alpha=0.7, color='lightcoral')
            ax.set_xlabel('Circuit Index'); ax.set_ylabel('Selectivity (CV of responses)'); ax.set_title('Circuit Selectivity to Classes'); ax.grid(True, alpha=0.3)
    else: ax.text(0.5, 0.5, "No classification or circuits.", ha='center', va='center', transform=ax.transAxes)
    ax = axes[0, 2]
    if circuit_activations.shape[1] >= 2: 
        pca_hidden_brain = PCA(n_components=2); circuit_pca = pca_hidden_brain.fit_transform(circuit_activations)
        scatter_colors_brain = test_Y_clf_labels_subset.flatten() if learner.n_outputs_classification > 0 else None
        cmap_brain = 'viridis' if learner.n_outputs_classification == 0 or scatter_colors_brain is None else 'tab10'
        scatter = ax.scatter(circuit_pca[:, 0], circuit_pca[:, 1], c=scatter_colors_brain, cmap=cmap_brain, alpha=0.6, s=15)
        if learner.n_outputs_classification > 0 and scatter_colors_brain is not None: plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f'PC1 ({pca_hidden_brain.explained_variance_ratio_[0]:.1%} var)'); ax.set_ylabel(f'PC2 ({pca_hidden_brain.explained_variance_ratio_[1]:.1%} var)'); ax.set_title('Population Vector (PCA)')
    else: ax.text(0.5, 0.5, "Need >= 2 circuits for PCA.", ha='center', va='center', transform=ax.transAxes)
    ax = axes[1, 0]; epochs_plot = learner.training_history['epoch']
    activation_evolution = np.array(learner.training_history['circuit_activations_mean_epoch'])
    if epochs_plot and activation_evolution.ndim == 2 and activation_evolution.shape[1] > 0:
        n_circuits_to_show = min(5, activation_evolution.shape[1]); circuit_indices = np.linspace(0, activation_evolution.shape[1]-1, n_circuits_to_show, dtype=int)
        for i, circuit_idx in enumerate(circuit_indices):
            ax.plot(epochs_plot, activation_evolution[:, circuit_idx], label=f'Circuit {circuit_idx}', alpha=0.8, linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Activation'); ax.set_title('Circuit Activation Evolution'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3)
    else: ax.text(0.5, 0.5, "No activation history.", ha='center', va='center', transform=ax.transAxes)
    ax = axes[1, 1]
    if circuit_activations.shape[1] > 1: 
        circuit_corr = np.corrcoef(circuit_activations.T); mask = np.triu(np.ones_like(circuit_corr, dtype=bool), k=1); correlations_brain = circuit_corr[mask]
        if correlations_brain.size > 0:
            sns.histplot(correlations_brain, bins=20, kde=True, ax=ax, color='lightgreen')
            ax.axvline(np.mean(correlations_brain), color='red', linestyle='--', label=f'Mean: {np.mean(correlations_brain):.2f}')
            ax.set_xlabel('Pairwise Correlation'); ax.set_ylabel('Density'); ax.set_title('Circuit Redundancy'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3)
        else: ax.text(0.5, 0.5, "Not enough data for correlations.", ha='center', va='center', transform=ax.transAxes)
    else: ax.text(0.5, 0.5, "Need >1 circuit for redundancy.", ha='center', va='center', transform=ax.transAxes)
    ax = axes[1, 2]
    if circuit_activations.shape[1] > 0:
        info_efficiency_proxy = np.var(circuit_activations, axis=0)
        ax.bar(range(len(info_efficiency_proxy)), info_efficiency_proxy, alpha=0.7, color='gold')
        ax.set_xlabel('Circuit Index'); ax.set_ylabel('Activation Variance (Proxy)'); ax.set_title('Circuit "Information" (Variance)'); ax.grid(True, alpha=0.3)
    else: ax.text(0.5, 0.5, "No circuits for info efficiency.", ha='center', va='center', transform=ax.transAxes)
    plt.suptitle('Brain-like Properties Analysis (Test Data Subset)', fontsize=16, y=0.98); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
    print("\n🧠 Brain-like Properties Summary (Test Data Subset):")
    if 'sparsity_levels' in locals() and sparsity_levels: print(f"Average Sparsity (<{threshold_sparsity} act.): {np.mean(sparsity_levels):.3f}")
    if 'selectivity_scores' in locals() and selectivity_scores: print(f"Average Selectivity (CV): {np.mean(selectivity_scores):.3f}")
    if 'correlations_brain' in locals() and correlations_brain.size > 0: print(f"Average Circuit Correlation: {np.mean(correlations_brain):.3f}")
    if 'pca_hidden_brain' in locals() and hasattr(pca_hidden_brain, 'explained_variance_ratio_') and pca_hidden_brain.explained_variance_ratio_.size >=2 : print(f"Population Variance Explained (PC1+PC2): {np.sum(pca_hidden_brain.explained_variance_ratio_[:2]):.1%}")

def plot_learning_dynamics(learner):
    if not learner.training_history['epoch']: print("Skipping learning dynamics plot: No training history."); return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10)); fig.subplots_adjust(hspace=0.35, wspace=0.25)
    epochs_plot = learner.training_history['epoch']
    ax = axes[0, 0]
    if learner.n_outputs_regression > 0: ax.plot(epochs_plot, learner.training_history['loss_reg'], label='Reg Loss (Train)', linewidth=1.5)
    if learner.n_outputs_classification > 0: ax.plot(epochs_plot, learner.training_history['loss_clf'], label='Clf Loss (Train)', linewidth=1.5)
    if learner.l1_activation_lambda > 0 and 'loss_l1_act' in learner.training_history and learner.training_history['loss_l1_act']:
        ax.plot(epochs_plot, learner.training_history['loss_l1_act'], label='L1 Act Loss (Train)', linewidth=1.5, linestyle=':')
    ax.plot(epochs_plot, learner.training_history['loss'], label='Total Loss (Train)', linewidth=2, linestyle='-')
    if learner.training_history['val_loss'] and len(learner.training_history['val_loss']) == len(epochs_plot):
        if learner.n_outputs_regression > 0: ax.plot(epochs_plot, learner.training_history['val_loss_reg'], label='Reg Loss (Val)', linestyle='--', linewidth=1.5)
        if learner.n_outputs_classification > 0: ax.plot(epochs_plot, learner.training_history['val_loss_clf'], label='Clf Loss (Val)', linestyle='--', linewidth=1.5)
        ax.plot(epochs_plot, learner.training_history['val_loss'], label='Total Loss (Val)', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (Log Scale)'); ax.set_title('Loss Component Evolution'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    ax = axes[0, 1]
    if learner.training_history['gradient_norms_w1']: ax.plot(epochs_plot, learner.training_history['gradient_norms_w1'], label='Grad W1', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_regression > 0 and learner.training_history['gradient_norms_w2_reg']: ax.plot(epochs_plot, learner.training_history['gradient_norms_w2_reg'], label='Grad W2_Reg', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_classification > 0 and learner.training_history['gradient_norms_w2_clf']: ax.plot(epochs_plot, learner.training_history['gradient_norms_w2_clf'], label='Grad W2_Clf', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient Norm (Log Scale)'); ax.set_title('Gradient Magnitude Evolution'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3)
    if any(learner.training_history['gradient_norms_w1']) or \
       (learner.n_outputs_regression > 0 and any(learner.training_history['gradient_norms_w2_reg'])) or \
       (learner.n_outputs_classification > 0 and any(learner.training_history['gradient_norms_w2_clf'])):
        ax.set_yscale('log', nonpositive='clip') 
    ax = axes[1, 0]
    if learner.training_history['weight_changes_w1']: ax.plot(epochs_plot, learner.training_history['weight_changes_w1'], label='ΔW1', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_regression > 0 and learner.training_history['weight_changes_w2_reg']: ax.plot(epochs_plot, learner.training_history['weight_changes_w2_reg'], label='ΔW2_Reg', alpha=0.8, linewidth=1.5)
    if learner.n_outputs_classification > 0 and learner.training_history['weight_changes_w2_clf']: ax.plot(epochs_plot, learner.training_history['weight_changes_w2_clf'], label='ΔW2_Clf', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Weight Change Magnitude (Log Scale)'); ax.set_title('Weight Update Dynamics'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3)
    if any(learner.training_history['weight_changes_w1']) or \
       (learner.n_outputs_regression > 0 and any(learner.training_history['weight_changes_w2_reg'])) or \
       (learner.n_outputs_classification > 0 and any(learner.training_history['weight_changes_w2_clf'])):
        ax.set_yscale('log', nonpositive='clip')
    ax = axes[1, 1]
    if len(epochs_plot) > 1:
        loss_changes = -np.diff(learner.training_history['loss']) 
        total_weight_change_epoch = np.zeros_like(loss_changes, dtype=float)
        if learner.training_history['weight_changes_w1'] and len(learner.training_history['weight_changes_w1']) > 1:
            total_weight_change_epoch += np.array(learner.training_history['weight_changes_w1'][1:])
        if learner.n_outputs_regression > 0 and learner.training_history['weight_changes_w2_reg'] and len(learner.training_history['weight_changes_w2_reg']) > 1:
            total_weight_change_epoch += np.array(learner.training_history['weight_changes_w2_reg'][1:])
        if learner.n_outputs_classification > 0 and learner.training_history['weight_changes_w2_clf'] and len(learner.training_history['weight_changes_w2_clf']) > 1:
            total_weight_change_epoch += np.array(learner.training_history['weight_changes_w2_clf'][1:])
        if total_weight_change_epoch.size == loss_changes.size: 
            efficiency = loss_changes / (total_weight_change_epoch + 1e-8) 
            window_size = min(len(efficiency), max(1, len(epochs_plot) // 20))
            if window_size > 1 and len(efficiency) >= window_size:
                efficiency_smoothed = np.convolve(efficiency, np.ones(window_size)/window_size, mode='valid')
                ax.plot(epochs_plot[1:len(efficiency_smoothed)+1], efficiency_smoothed, color='darkcyan', linewidth=2)
            elif len(efficiency) > 0 : 
                 ax.plot(epochs_plot[1:len(efficiency)+1], efficiency, color='darkcyan', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Efficiency (ΔLoss / ΔWeight)'); ax.set_title('Learning Efficiency (Smoothed)'); ax.grid(True, alpha=0.3); ax.axhline(0, color='grey', linestyle=':', linewidth=1) 
    plt.suptitle('Learning Dynamics Analysis', fontsize=16, y=0.98); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

# --- More Complex Synthetic Data Generation ---
def generate_synthetic_mtl_data_complex(
    n_samples: int = 1000, n_features: int = 5, seed: int = 42, 
    noise_level_reg: float = 0.15, noise_level_clf: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    
    inputs = np.zeros((n_samples, n_features))
    inputs[:, 0] = np.random.uniform(-2, 2, n_samples) # x0
    inputs[:, 1] = np.random.uniform(-2, 2, n_samples) # x1
    inputs[:, 2] = np.random.normal(0, 1, n_samples)   # x2
    inputs[:, 3] = np.sin(np.pi * inputs[:, 0]) + np.cos(0.5 * np.pi * inputs[:,1]) # x3 (derived)
    inputs[:, 4] = inputs[:, 0] * inputs[:, 1] * np.exp(-0.2 * inputs[:, 2]**2) # x4 (interaction)

    # Task 1: Complex Regression
    target_regression = (
        0.7 * np.sin(1.5 * inputs[:, 0]) * np.cos(1.2 * inputs[:, 1]) +
        0.5 * np.tanh(0.8 * inputs[:, 2] - 0.5 * inputs[:, 3]) +
        0.3 * inputs[:, 4] * np.cos(inputs[:,0]) + # Use interaction term
        0.2 * inputs[:, 1]**2 * np.sin(inputs[:,2]) -
        0.1 * inputs[:,0] * inputs[:,3] +
        np.random.normal(0, noise_level_reg, n_samples)
    )
    target_regression = (target_regression - np.mean(target_regression)) / (np.std(target_regression) + 1e-6)
    target_regression = target_regression.reshape(-1, 1)

    # Task 2: Complex Classification (3 classes)
    target_classification_labels = np.zeros(n_samples, dtype=int)
    
    # Class 0: Central ellipsoid + condition on x3
    cond0_base = (inputs[:, 0]**2 / 2.0 + inputs[:, 1]**2 / 1.5 + inputs[:, 2]**2 / 3.0 < 1.0)
    cond0 = cond0_base & (inputs[:, 3] > 0.2)
    
    # Class 1: "Twisted sheet" or region based on x0, x1, x4
    cond1_base = (np.abs(inputs[:, 0] + np.sin(inputs[:, 1]*2)) < 0.8) & (inputs[:, 4] < np.mean(inputs[:,4]) - 0.5*np.std(inputs[:,4]))
    cond1 = cond1_base & ~cond0
    
    # Class 2: Default, or specific outer regions
    # All remaining points initially go to class 2
    target_classification_labels.fill(2) 
    target_classification_labels[cond1] = 1
    target_classification_labels[cond0] = 0 # Apply class 0 last to ensure it overrides if overlaps

    # Add more specific rules for class 2 to make it distinct
    cond2_specific = (inputs[:,2] > 1.5) & (inputs[:,0] < -1.0) & ~cond0 & ~cond1
    target_classification_labels[cond2_specific] = 2


    # Add label noise
    num_flips = int(noise_level_clf * n_samples)
    flip_indices = np.random.choice(n_samples, num_flips, replace=False)
    for idx in flip_indices:
        current_label = target_classification_labels[idx]
        possible_new_labels = [l for l in [0, 1, 2] if l != current_label]
        if possible_new_labels:
            target_classification_labels[idx] = np.random.choice(possible_new_labels)

    target_classification = target_classification_labels.reshape(-1, 1)
    
    return inputs, target_regression, target_classification

# --- Metrics (remain same) ---
def r_squared(y_true, y_pred):
    if y_true.size == 0 or y_pred.size == 0: return 0.0
    ss_res = np.sum((y_true - y_pred)**2); ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2)
    if ss_tot == 0: return 1.0 if ss_res < 1e-9 else 0.0
    return 1.0 - (ss_res / ss_tot)

def accuracy_score(y_true_labels, y_pred_probs):
    if y_true_labels.size == 0 or y_pred_probs.size == 0: return 0.0
    if y_pred_probs.shape[1] == 0 : return 0.0
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    return np.mean(y_true_labels.flatten() == y_pred_labels.flatten())

# --- Main Execution (Adjusted for More Complex Task) ---
if __name__ == "__main__":
    print("🧠 Multi-Task Complex Learner - MORE COMPLEX DATA Demo")
    print("=" * 70)

    NUM_SAMPLES = 5000 # Increased samples
    TRAIN_SPLIT_RATIO = 0.8
    VALIDATION_SPLIT_TRAIN = 0.20 
    
    N_HIDDEN_CIRCUITS = 30   # Increased capacity
    N_INTERNAL_UNITS = 6    # Increased capacity
    LEARNING_RATE = 0.002   # Potentially smaller for stability
    N_EPOCHS = 6000         # More epochs
    BATCH_SIZE = 64        
    CLIP_GRAD_VAL = 1.0

    MIN_EPOCHS_NO_IMPROVE = 100 
    PATIENCE_NO_IMPROVE = 200   
    
    RANDOM_SEED_DATA = 2027
    RANDOM_SEED_NETWORK = 104

    MICROCIRCUIT_AGGREGATION = "max"  
    L1_ACTIVATION_LAMBDA = 0.00005 # Start small for complex data
    USE_KWTA_ON_CIRCUITS = True      
    KWTA_K_CIRCUITS = 10 # ~1/3 of circuits active

    LOSS_WEIGHT_REGRESSION = 1.0
    LOSS_WEIGHT_CLASSIFICATION = 1.0 

    N_FEATURES_COMPLEX = 5
    N_OUTPUTS_REGRESSION = 1 
    N_OUTPUTS_CLASSIFICATION = 3 # 3 classes
    
    inputs, targets_reg, targets_clf_labels = generate_synthetic_mtl_data_complex(
        NUM_SAMPLES, n_features=N_FEATURES_COMPLEX, seed=RANDOM_SEED_DATA, 
        noise_level_reg=0.15, noise_level_clf=0.1
    )
    
    n_train_val = int(NUM_SAMPLES * TRAIN_SPLIT_RATIO)
    train_val_X, train_val_Y_reg, train_val_Y_clf_labels = \
        inputs[:n_train_val], targets_reg[:n_train_val], targets_clf_labels[:n_train_val]
    test_X, test_Y_reg, test_Y_clf_labels = \
        inputs[n_train_val:], targets_reg[n_train_val:], targets_clf_labels[n_train_val:]

    print(f"Generated {NUM_SAMPLES} samples with {inputs.shape[1]} input features.")
    print(f"Training/Validation data shapes: X={train_val_X.shape}, Y_reg={train_val_Y_reg.shape}, Y_clf={train_val_Y_clf_labels.shape}")
    print(f"Testing data shapes: X={test_X.shape}, Y_reg={test_Y_reg.shape}, Y_clf={test_Y_clf_labels.shape}")
    if N_OUTPUTS_CLASSIFICATION > 0:
        print(f"Class distribution in training: {np.bincount(train_val_Y_clf_labels.flatten())}")
        if test_X.shape[0] > 0: print(f"Class distribution in test: {np.bincount(test_Y_clf_labels.flatten())}")

    n_inputs_actual = train_val_X.shape[1]
    
    learner = MultiTaskComplexLearner(
        n_inputs=n_inputs_actual,
        n_outputs_regression=N_OUTPUTS_REGRESSION,
        n_outputs_classification=N_OUTPUTS_CLASSIFICATION,
        n_hidden_circuits=N_HIDDEN_CIRCUITS,
        n_internal_units_per_circuit=N_INTERNAL_UNITS,
        learning_rate=LEARNING_RATE,
        loss_weight_regression=LOSS_WEIGHT_REGRESSION,
        loss_weight_classification=LOSS_WEIGHT_CLASSIFICATION,
        microcircuit_aggregation=MICROCIRCUIT_AGGREGATION,
        l1_activation_lambda=L1_ACTIVATION_LAMBDA,
        use_kwta_on_circuits=USE_KWTA_ON_CIRCUITS,
        kwta_k_circuits=KWTA_K_CIRCUITS,
        seed=RANDOM_SEED_NETWORK
    )
    print(f"\nInitialized MultiTaskComplexLearner with {learner.n_hidden_circuits} hidden circuits ({learner.n_internal_units_per_circuit} internal units each).")
    print(f"Activation: ReLU, Micro-Agg: {MICROCIRCUIT_AGGREGATION}, L1 Lambda: {L1_ACTIVATION_LAMBDA}")
    print(f"kWTA on Circuits: {USE_KWTA_ON_CIRCUITS}, k: {KWTA_K_CIRCUITS if USE_KWTA_ON_CIRCUITS else 'N/A'}")
    print(f"Input dims: {n_inputs_actual}, Reg Outputs: {N_OUTPUTS_REGRESSION}, Clf Classes: {N_OUTPUTS_CLASSIFICATION}")

    print("\n🚀 Starting Training...")
    import time
    start_time = time.time()
    learner.learn(
        train_val_X, train_val_Y_reg, train_val_Y_clf_labels,
        n_epochs=N_EPOCHS,
        min_epochs_no_improve=MIN_EPOCHS_NO_IMPROVE,
        patience_no_improve=PATIENCE_NO_IMPROVE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT_TRAIN, 
        verbose=True,
        clip_val=CLIP_GRAD_VAL
    )
    end_time = time.time()
    print(f"--- Training finished in {end_time - start_time:.2f} seconds ---")
    
    title_prefix_plots = f"ComplexMTL Agg={MICROCIRCUIT_AGGREGATION}, L1={L1_ACTIVATION_LAMBDA}, kWTA={USE_KWTA_ON_CIRCUITS}(k={KWTA_K_CIRCUITS if USE_KWTA_ON_CIRCUITS else 'N/A'}) "
    
    plot_comprehensive_analysis(learner, test_X, test_Y_reg, test_Y_clf_labels, title_prefix=title_prefix_plots)
    plot_learning_dynamics(learner)
    if test_X.shape[0] > 0:
        plot_brain_like_analysis(learner, test_X, test_Y_reg, test_Y_clf_labels)
    else:
        print("Skipping brain-like analysis plot as there is no test data.")

    print("\n🧪 Evaluating on Test Data (Console Summary)...")
    if test_X.shape[0] > 0:
        test_preds_reg, test_preds_clf_probs, _ = learner.predict(test_X, batch_size=BATCH_SIZE)
        
        if N_OUTPUTS_REGRESSION > 0:
            mse_test_reg = np.mean((test_Y_reg - test_preds_reg) ** 2)
            mae_test_reg = np.mean(np.abs(test_Y_reg - test_preds_reg))
            r2_test_reg = r_squared(test_Y_reg, test_preds_reg)
            print("\n--- Regression Task ---")
            print(f"Test Mean Squared Error (MSE): {mse_test_reg:.6f}")
            print(f"Test Mean Absolute Error (MAE): {mae_test_reg:.6f}")
            print(f"Test R-squared:                {r2_test_reg:.6f}")

        if N_OUTPUTS_CLASSIFICATION > 0:
            acc_test_clf = accuracy_score(test_Y_clf_labels, test_preds_clf_probs)
            print("\n--- Classification Task ---")
            print(f"Test Accuracy:                 {acc_test_clf:.6f}")
            
            try:
                from sklearn.metrics import classification_report
                pred_labels = np.argmax(test_preds_clf_probs, axis=1)
                print("\nClassification Report (Test Data):")
                print(classification_report(test_Y_clf_labels.flatten(), pred_labels, 
                                            target_names=[f"Class {i}" for i in range(N_OUTPUTS_CLASSIFICATION)],
                                            zero_division=0))
            except ImportError:
                print("Install scikit-learn to see the classification report.")
    else:
        print("No test data to evaluate for console summary.")
    print("\n✅ Multi-Task Brain-like (Complex Data) Demo Completed.")
