import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Callable
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from numba import jit
import warnings
from matplotlib.animation import FuncAnimation # Added for animation

warnings.filterwarnings('ignore')

# === NEURAL NETWORK COMPONENTS ===

@jit(nopython=True, cache=True)
def relu_numba(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

@jit(nopython=True, cache=True)
def relu_derivative_numba(x_activated: np.ndarray) -> np.ndarray:
    return (x_activated > 0.0).astype(x_activated.dtype)

@jit(nopython=True, cache=True)
def sigmoid_numba(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

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
    for i in range(indices.shape[0]):
        result[indices[i]] = activations[indices[i]]
    return result

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

@jit(nopython=True, cache=True)
def _backward_pass_static_jitted(
        pred_reg_np, target_reg_np,
        pred_clf_sigmoid_probs_np, target_clf_binary_labels_np,
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
        internal_activations_i = all_internal_activations_sample_np[i, :]
        if internal_activations_i.size > 0:
            ds_dz_internal = relu_derivative_numba(internal_activations_i)
            circuit_derivative = 0.0
            if microcircuit_aggregation_method_code == 0: # mean
                if internal_activations_i.shape[0] > 0: # Avoid division by zero for mean
                    circuit_derivative = np.sum(ds_dz_internal * hidden_circuits_iw_np[i, :]) / internal_activations_i.shape[0]
                else:
                    circuit_derivative = 0.0
            elif microcircuit_aggregation_method_code == 1: # max
                idx_max = np.argmax(internal_activations_i)
                circuit_derivative = ds_dz_internal[idx_max] * hidden_circuits_iw_np[i, idx_max]
            dL_dz_h_shared[i] = total_error_propagated_to_hidden_outputs[i] * circuit_derivative
        else:
            dL_dz_h_shared[i] = 0.0

    dW1 = np.outer(x_np, dL_dz_h_shared)
    db1 = dL_dz_h_shared

    return dW1, db1, dW2_reg, db2_reg, dW2_clf, db2_clf

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
                 l1_activation_lambda: float = 0.0,
                 use_kwta_on_circuits: bool = False,
                 kwta_k_circuits: int = 0,
                 seed: int = 42):
        np.random.seed(seed)
        self.n_inputs = n_inputs
        self.n_outputs_regression = n_outputs_regression
        self.n_outputs_classification = n_binary_clf_tasks
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

        if use_kwta_on_circuits and (kwta_k_circuits <= 0 or kwta_k_circuits > n_hidden_circuits):
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

        if self.n_outputs_classification > 0:
            limit_w2_clf = np.sqrt(6.0 / (n_hidden_circuits + self.n_outputs_classification))
            self.W2_clf = np.random.uniform(-limit_w2_clf, limit_w2_clf, (n_hidden_circuits, self.n_outputs_classification)).astype(np.float64)
            self.b2_clf = np.zeros(self.n_outputs_classification, dtype=np.float64)
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
                if microcircuit_aggregation_method_code == 0: # mean
                    a_h_shared_pre_kwta[i] = np.mean(internal_activations)
                elif microcircuit_aggregation_method_code == 1: # max
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
        prediction_clf_sigmoid_probs = np.empty(n_outputs_classification, dtype=np.float64)
        if n_outputs_classification > 0:
            logits_clf = np.dot(a_h_shared, W2_clf_np) + b2_clf_np
            prediction_clf_sigmoid_probs = sigmoid_numba(logits_clf)

        return prediction_reg, prediction_clf_sigmoid_probs, logits_clf, a_h_shared, all_internal_activations_sample

    def forward_pass(self, input_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        x_np = input_params.astype(np.float64)
        pred_reg, pred_clf_sigmoid_probs, logits_clf, a_h_shared, all_internal_acts_sample = \
            self._forward_pass_static(
                x_np, self.W1, self.b1,
                self.W2_reg, self.b2_reg, self.W2_clf, self.b2_clf,
                self.hidden_circuits_internal_weights, self.hidden_circuits_internal_biases,
                self.n_hidden_circuits, self.n_internal_units_per_circuit,
                self.n_outputs_regression, self.n_outputs_classification,
                self.microcircuit_aggregation_code,
                self.use_kwta_on_circuits, self.kwta_k_circuits
            )
        cache = {
            'x': x_np, 'hidden_circuit_outputs_shared': a_h_shared,
            'all_internal_activations_sample': all_internal_acts_sample,
            'prediction_reg': pred_reg,
            'prediction_clf_sigmoid_probs': pred_clf_sigmoid_probs,
            'logits_clf': logits_clf,
        }
        return pred_reg, pred_clf_sigmoid_probs, cache

    def backward_pass_to_get_grads(self,
                                   target_reg: np.ndarray,
                                   target_clf_binary_labels: np.ndarray,
                                   cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_np = cache['x']
        a_h_shared_np = cache['hidden_circuit_outputs_shared']
        all_internal_acts_sample_np = cache['all_internal_activations_sample']
        pred_reg_np = cache['prediction_reg']
        pred_clf_sigmoid_probs_np = cache['prediction_clf_sigmoid_probs']

        target_reg_np = target_reg.astype(np.float64)
        target_clf_binary_labels_np = target_clf_binary_labels.astype(np.float64)

        eff_pred_reg_np = pred_reg_np if self.n_outputs_regression > 0 else np.empty(0,dtype=np.float64)
        eff_target_reg_np = target_reg_np if self.n_outputs_regression > 0 else np.empty(0,dtype=np.float64)
        eff_pred_clf_sigmoid_probs_np = pred_clf_sigmoid_probs_np if self.n_outputs_classification > 0 else np.empty(0,dtype=np.float64)
        eff_target_clf_binary_labels_np = target_clf_binary_labels_np if self.n_outputs_classification > 0 else np.empty(0,dtype=np.float64)

        return _backward_pass_static_jitted(
            eff_pred_reg_np, eff_target_reg_np,
            eff_pred_clf_sigmoid_probs_np, eff_target_clf_binary_labels_np,
            x_np, a_h_shared_np, all_internal_acts_sample_np,
            self.W1, self.W2_reg, self.W2_clf, self.hidden_circuits_internal_weights,
            self.n_hidden_circuits, self.n_internal_units_per_circuit,
            self.loss_weight_regression if self.n_outputs_regression > 0 else 0.0,
            self.loss_weight_classification if self.n_outputs_classification > 0 else 0.0,
            self.microcircuit_aggregation_code, self.l1_activation_lambda
        )

    def learn(self, input_data: np.ndarray, target_data_reg: np.ndarray,
               target_data_clf_binary_labels: np.ndarray,
               n_epochs: int = 1000, min_epochs_no_improve: int = 50,
               patience_no_improve: int = 100, batch_size: int = 32,
               validation_split: float = 0.1, verbose: bool = True, clip_val: float = 1.0):
        n_samples = input_data.shape[0]

        if validation_split > 0 and n_samples * validation_split >= 1:
            val_size = int(n_samples * validation_split)
            permutation = np.random.permutation(n_samples)
            val_input = input_data[permutation[:val_size]]
            val_target_reg = target_data_reg[permutation[:val_size]]
            val_target_clf_binary = target_data_clf_binary_labels[permutation[:val_size]]
            train_input = input_data[permutation[val_size:]]
            train_target_reg = target_data_reg[permutation[val_size:]]
            train_target_clf_binary = target_data_clf_binary_labels[permutation[val_size:]]
            n_train_samples = train_input.shape[0]
            if verbose:
                print(f"Training on {n_train_samples} samples, validating on {val_size} samples.")
        else:
            train_input, train_target_reg, train_target_clf_binary = input_data, target_data_reg, target_data_clf_binary_labels
            val_input, val_target_reg, val_target_clf_binary = None, None, None
            n_train_samples = train_input.shape[0]
            if verbose:
                print(f"Training on {n_train_samples} samples (no validation split).")

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
            shuffled_train_target_clf_binary = train_target_clf_binary[permutation]

            for i in range(0, n_train_samples, batch_size):
                batch_input = shuffled_train_input[i:i+batch_size]
                batch_target_reg = shuffled_train_target_reg[i:i+batch_size]
                batch_target_clf_binary_b = shuffled_train_target_clf_binary[i:i+batch_size]
                current_batch_size = batch_input.shape[0]

                batch_dW1 = np.zeros_like(self.W1)
                batch_db1 = np.zeros_like(self.b1)
                batch_dW2_reg = np.zeros_like(self.W2_reg) if self.n_outputs_regression > 0 else None
                batch_db2_reg = np.zeros_like(self.b2_reg) if self.n_outputs_regression > 0 else None
                batch_dW2_clf = np.zeros_like(self.W2_clf) if self.n_outputs_classification > 0 else None
                batch_db2_clf = np.zeros_like(self.b2_clf) if self.n_outputs_classification > 0 else None

                for j_sample in range(current_batch_size):
                    single_input = batch_input[j_sample]
                    single_target_reg = batch_target_reg[j_sample] if self.n_outputs_regression > 0 else np.empty(0)
                    single_target_clf_binary_s = batch_target_clf_binary_b[j_sample] if self.n_outputs_classification > 0 else np.empty(0)

                    pred_reg, pred_clf_sigmoid_probs, cache = self.forward_pass(single_input)

                    epoch_circuit_activations_list.append(cache['hidden_circuit_outputs_shared'].copy())

                    loss_reg_sample = 0.0
                    if self.n_outputs_regression > 0 and single_target_reg.size > 0 and pred_reg.size > 0 :
                        loss_reg_sample = np.mean((single_target_reg - pred_reg) ** 2)


                    loss_clf_sample = 0.0
                    if self.n_outputs_classification > 0 and single_target_clf_binary_s.size > 0 and cache['logits_clf'].size > 0:
                        loss_clf_sample = binary_cross_entropy_numba_multi_task(
                            single_target_clf_binary_s,
                            cache['logits_clf']
                        )

                    l1_act_penalty_sample = self.l1_activation_lambda * np.sum(np.abs(cache['hidden_circuit_outputs_shared'])) if self.l1_activation_lambda > 0 else 0.0
                    total_loss_sample = (self.loss_weight_regression*loss_reg_sample) + (self.loss_weight_classification*loss_clf_sample) + l1_act_penalty_sample

                    epoch_train_loss_total += total_loss_sample
                    epoch_train_loss_reg += loss_reg_sample
                    epoch_train_loss_clf += loss_clf_sample
                    epoch_train_loss_l1_act += l1_act_penalty_sample

                    eff_target_reg = single_target_reg if self.n_outputs_regression > 0 else np.empty((0,),dtype=np.float64)
                    eff_target_clf_binary = single_target_clf_binary_s if self.n_outputs_classification > 0 else np.empty((0,),dtype=np.float64)

                    dW1_s,db1_s,dW2_reg_s,db2_reg_s,dW2_clf_s,db2_clf_s = self.backward_pass_to_get_grads(eff_target_reg, eff_target_clf_binary, cache)

                    batch_dW1 += dW1_s
                    batch_db1 += db1_s
                    if self.n_outputs_regression > 0 and dW2_reg_s is not None:
                        batch_dW2_reg += dW2_reg_s
                        batch_db2_reg += db2_reg_s
                    if self.n_outputs_classification > 0 and dW2_clf_s is not None:
                        batch_dW2_clf += dW2_clf_s
                        batch_db2_clf += db2_clf_s

                avg_dW1 = batch_dW1/current_batch_size
                avg_db1 = batch_db1/current_batch_size
                epoch_grad_norms_w1_list.append(np.linalg.norm(avg_dW1))

                self.W1 -= self.learning_rate*np.clip(avg_dW1,-clip_val,clip_val)
                self.b1 -= self.learning_rate*np.clip(avg_db1,-clip_val,clip_val)

                if self.n_outputs_regression > 0 and batch_dW2_reg is not None:
                    avg_dW2_reg = batch_dW2_reg/current_batch_size
                    avg_db2_reg = batch_db2_reg/current_batch_size
                    epoch_grad_norms_w2_reg_list.append(np.linalg.norm(avg_dW2_reg))
                    self.W2_reg -= self.learning_rate*np.clip(avg_dW2_reg,-clip_val,clip_val)
                    self.b2_reg -= self.learning_rate*np.clip(avg_db2_reg,-clip_val,clip_val)

                if self.n_outputs_classification > 0 and batch_dW2_clf is not None:
                    avg_dW2_clf = batch_dW2_clf/current_batch_size
                    avg_db2_clf = batch_db2_clf/current_batch_size
                    epoch_grad_norms_w2_clf_list.append(np.linalg.norm(avg_dW2_clf))
                    self.W2_clf -= self.learning_rate*np.clip(avg_dW2_clf,-clip_val,clip_val)
                    self.b2_clf -= self.learning_rate*np.clip(avg_db2_clf,-clip_val,clip_val)

            weight_change_w1 = np.linalg.norm(self.W1-prev_W1)
            weight_change_w2_reg = np.linalg.norm(self.W2_reg-prev_W2_reg) if self.n_outputs_regression > 0 and prev_W2_reg is not None else 0.0
            weight_change_w2_clf = np.linalg.norm(self.W2_clf-prev_W2_clf) if self.n_outputs_classification > 0 and prev_W2_clf is not None else 0.0

            avg_epoch_train_loss = epoch_train_loss_total/n_train_samples if n_train_samples > 0 else 0
            avg_epoch_train_loss_reg = epoch_train_loss_reg/n_train_samples if n_train_samples > 0 else 0
            avg_epoch_train_loss_clf = epoch_train_loss_clf/n_train_samples if n_train_samples > 0 else 0
            avg_epoch_train_loss_l1_act = epoch_train_loss_l1_act/n_train_samples if n_train_samples > 0 else 0

            self.training_history['loss'].append(avg_epoch_train_loss)
            self.training_history['loss_reg'].append(avg_epoch_train_loss_reg)
            self.training_history['loss_clf'].append(avg_epoch_train_loss_clf)
            self.training_history['loss_l1_act'].append(avg_epoch_train_loss_l1_act)
            self.training_history['epoch'].append(epoch)
            self.training_history['learning_rate'].append(self.learning_rate)
            self.training_history['gradient_norms_w1'].append(np.mean(epoch_grad_norms_w1_list) if epoch_grad_norms_w1_list else 0)
            self.training_history['gradient_norms_w2_reg'].append(np.mean(epoch_grad_norms_w2_reg_list) if epoch_grad_norms_w2_reg_list else 0)
            self.training_history['gradient_norms_w2_clf'].append(np.mean(epoch_grad_norms_w2_clf_list) if epoch_grad_norms_w2_clf_list else 0)
            self.training_history['circuit_activations_mean_epoch'].append(np.mean(epoch_circuit_activations_list,axis=0) if epoch_circuit_activations_list else np.zeros(self.n_hidden_circuits))
            self.training_history['weight_changes_w1'].append(weight_change_w1)
            self.training_history['weight_changes_w2_reg'].append(weight_change_w2_reg)
            self.training_history['weight_changes_w2_clf'].append(weight_change_w2_clf)

            current_val_loss_total = avg_epoch_train_loss
            current_val_loss_reg = avg_epoch_train_loss_reg
            current_val_loss_clf = avg_epoch_train_loss_clf

            if val_input is not None and len(val_input) > 0:
                val_preds_reg, val_preds_clf_sigmoid_probs, val_logits_clf_val = self.predict(val_input, batch_size=batch_size)
                current_val_loss_reg_unweighted = 0.0
                if self.n_outputs_regression > 0 and val_target_reg.size > 0 and val_preds_reg.size > 0:
                    current_val_loss_reg_unweighted = np.mean((val_target_reg - val_preds_reg)**2)
                current_val_loss_clf_unweighted = 0.0
                if self.n_outputs_classification > 0 and val_target_clf_binary.size > 0 and val_logits_clf_val.size > 0:
                     current_val_loss_clf_unweighted = binary_cross_entropy_numba_multi_task(val_target_clf_binary, val_logits_clf_val)
                current_val_loss_total = (self.loss_weight_regression * current_val_loss_reg_unweighted) + \
                                         (self.loss_weight_classification * current_val_loss_clf_unweighted)
                current_val_loss_reg = current_val_loss_reg_unweighted
                current_val_loss_clf = current_val_loss_clf_unweighted

            self.training_history['val_loss'].append(current_val_loss_total)
            self.training_history['val_loss_reg'].append(current_val_loss_reg)
            self.training_history['val_loss_clf'].append(current_val_loss_clf)

            log_interval = max(1, n_epochs//20) if n_epochs > 0 else 1
            if verbose and (epoch+1)%log_interval == 0:
                print(f"Epoch {epoch+1:4d}/{n_epochs}: Train Loss={avg_epoch_train_loss:.4f} (Reg: {avg_epoch_train_loss_reg:.4f}, Clf: {avg_epoch_train_loss_clf:.4f}, L1Act: {avg_epoch_train_loss_l1_act:.4f}), Val Loss={current_val_loss_total:.4f} (Reg: {current_val_loss_reg:.4f}, Clf: {current_val_loss_clf:.4f})")

            loss_for_stopping = current_val_loss_total
            if loss_for_stopping < best_val_loss:
                best_val_loss = loss_for_stopping
                epochs_without_improvement = 0
                best_weights = {'W1':self.W1.copy(),'b1':self.b1.copy()}
                if self.n_outputs_regression > 0:
                    best_weights['W2_reg']=self.W2_reg.copy()
                    best_weights['b2_reg']=self.b2_reg.copy()
                if self.n_outputs_classification > 0:
                    best_weights['W2_clf']=self.W2_clf.copy()
                    best_weights['b2_clf']=self.b2_clf.copy()
            else:
                epochs_without_improvement += 1

            if epoch >= min_epochs_no_improve and epochs_without_improvement >= patience_no_improve:
                if verbose: print(f"Early stopping at epoch {epoch+1}.")
                if best_weights:
                    self.W1,self.b1=best_weights['W1'],best_weights['b1']
                    if self.n_outputs_regression > 0 and 'W2_reg' in best_weights:
                        self.W2_reg,self.b2_reg=best_weights['W2_reg'],best_weights['b2_reg']
                    if self.n_outputs_classification > 0 and 'W2_clf' in best_weights:
                        self.W2_clf,self.b2_clf=best_weights['W2_clf'],best_weights['b2_clf']
                break
        if verbose: print(f"Training done! Best validation loss: {best_val_loss:.6f}")

    def predict(self, input_data: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = input_data.shape[0]
        predictions_reg = np.zeros((n_samples, self.n_outputs_regression)) if self.n_outputs_regression > 0 else np.empty((n_samples, 0))
        predictions_clf_sigmoid_probs = np.zeros((n_samples, self.n_outputs_classification)) if self.n_outputs_classification > 0 else np.empty((n_samples, 0))
        logits_clf_all = np.zeros((n_samples, self.n_outputs_classification)) if self.n_outputs_classification > 0 else np.empty((n_samples, 0))

        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0:
            effective_batch_size = n_samples
        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]
                current_batch_size_pred = batch_input.shape[0]
                for j_sample in range(current_batch_size_pred):
                    single_input = batch_input[j_sample]
                    pred_reg_s, pred_clf_sigmoid_probs_s, cache_s = self.forward_pass(single_input)
                    if self.n_outputs_regression > 0:
                        predictions_reg[i+j_sample, :] = pred_reg_s
                    if self.n_outputs_classification > 0:
                        predictions_clf_sigmoid_probs[i+j_sample, :] = pred_clf_sigmoid_probs_s
                        logits_clf_all[i+j_sample, :] = cache_s['logits_clf']
        return predictions_reg, predictions_clf_sigmoid_probs, logits_clf_all

    def get_circuit_activations(self, input_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        n_samples = input_data.shape[0]
        activations = np.zeros((n_samples, self.n_hidden_circuits))
        effective_batch_size = batch_size if n_samples >= batch_size else n_samples
        if effective_batch_size == 0 and n_samples > 0:
            effective_batch_size = n_samples
        if effective_batch_size > 0:
            for i in range(0, n_samples, effective_batch_size):
                batch_input = input_data[i:i+effective_batch_size]
                current_bs = batch_input.shape[0]
                for j_sample in range(current_bs):
                    single_input = batch_input[j_sample]
                    _, _, cache = self.forward_pass(single_input)
                    activations[i+j_sample] = cache['hidden_circuit_outputs_shared']
        return activations

# === RAT NAVIGATION COMPONENTS ===

class Action(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

@dataclass
class MazeState:
    position: Tuple[int, int]
    target_pattern: np.ndarray
    target_position: Tuple[int, int]
    visited_positions: List[Tuple[int, int]]
    steps_taken: int
    max_steps: int

class LabyrinthEnvironment:
    def __init__(self,
                 maze_size: Tuple[int, int] = (10, 10),
                 wall_density: float = 0.2,
                 n_patterns: int = 4,
                 pattern_dim: int = 8,
                 max_episode_steps: int = 100,
                 seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.maze_size = maze_size
        self.height, self.width = maze_size
        self.n_patterns = n_patterns
        self.pattern_dim = pattern_dim
        self.max_episode_steps = max_episode_steps
        self.maze = self._generate_maze(wall_density)
        self.patterns = self._generate_patterns()
        self.pattern_goals = self._assign_pattern_goals()
        self.current_state = None

    def _generate_maze(self, wall_density: float) -> np.ndarray:
        maze = np.zeros(self.maze_size, dtype=int)
        wall_positions = np.random.rand(*self.maze_size) < wall_density
        maze[wall_positions] = 1
        maze[0, :] = 1; maze[-1, :] = 1; maze[:, 0] = 1; maze[:, -1] = 1
        if self.width > 2: maze[0, self.width//2] = 0
        if self.width > 2: maze[-1, self.width//2] = 0
        if self.height > 2: maze[self.height//2, 0] = 0
        if self.height > 2: maze[self.height//2, -1] = 0
        center = (self.height//2, self.width//2)
        if 0 <= center[0] < self.height and 0 <= center[1] < self.width:
             maze[center] = 0
        return maze

    def _generate_patterns(self) -> np.ndarray:
        if self.n_patterns == 0: return np.empty((0, self.pattern_dim))
        patterns = np.random.randn(self.n_patterns, self.pattern_dim)
        for i in range(self.n_patterns):
            patterns[i, :i+1] = 2.0
            if self.pattern_dim > i+1 :
                 patterns[i, -(i+1):] = -2.0
        norm = np.linalg.norm(patterns, axis=1, keepdims=True)
        return patterns / (norm + 1e-9)

    def _assign_pattern_goals(self) -> Dict[int, List[Tuple[int, int]]]:
        free_positions = list(zip(*np.where(self.maze == 0)))
        center = (self.height//2, self.width//2)
        if center in free_positions: free_positions.remove(center)
        if not free_positions:
            print("Warning: No free positions for goals besides center. Making some border cells free.")
            # Ensure maze_size is large enough for border cells to be distinct from center/edges
            if self.height > 3 and self.width > 3:
                for r_idx in [1, self.height-2]:
                    for c_idx in [1, self.width-2]:
                        self.maze[r_idx,c_idx] = 0
            free_positions = list(zip(*np.where(self.maze == 0)))
            if center in free_positions: free_positions.remove(center)
            if not free_positions:
                if self.maze[center] == 0: free_positions.append(center) 
                else: raise ValueError("Cannot assign goals: No free positions available.")
        
        pattern_goals = {}
        if self.n_patterns == 0: return pattern_goals

        num_goals_to_assign = self.n_patterns
        assigned_goals_flat = [free_positions[i % len(free_positions)] for i in range(num_goals_to_assign)] if free_positions else []
        if not assigned_goals_flat and self.n_patterns > 0:
            raise ValueError("No goals could be assigned.")


        goals_per_pattern = max(1, len(assigned_goals_flat) // self.n_patterns)
        for i in range(self.n_patterns):
            start_idx = i * goals_per_pattern
            end_idx = (i + 1) * goals_per_pattern if i < self.n_patterns -1 else len(assigned_goals_flat)
            current_pattern_goals = assigned_goals_flat[start_idx:end_idx]
            if not current_pattern_goals: 
                current_pattern_goals = [assigned_goals_flat[i % len(assigned_goals_flat)]]
            pattern_goals[i] = current_pattern_goals
        return pattern_goals

    def reset(self, pattern_id: Optional[int] = None) -> Tuple[np.ndarray, MazeState]:
        if self.n_patterns == 0:
            print("Warning: Resetting environment with no patterns defined.")
            start_pos = (self.height//2, self.width//2)
            dummy_pattern_arr = np.zeros(self.pattern_dim)
            dummy_target_pos = start_pos
            self.current_state = MazeState(
                position=start_pos, target_pattern=dummy_pattern_arr,
                target_position=dummy_target_pos, visited_positions=[start_pos],
                steps_taken=0, max_steps=self.max_episode_steps)
            return self._get_observation(), self.current_state


        if pattern_id is None:
            pattern_id = np.random.randint(0, self.n_patterns)
        elif not (0 <= pattern_id < self.n_patterns):
            raise ValueError(f"Invalid pattern_id: {pattern_id}")

        start_pos = (self.height//2, self.width//2)
        if not (0 <= start_pos[0] < self.height and 0 <= start_pos[1] < self.width) or self.maze[start_pos] == 1 :
            free_cells_for_start = list(zip(*np.where(self.maze == 0)))
            if not free_cells_for_start: raise ValueError("No free cells to start.")
            start_pos = random.choice(free_cells_for_start)
        
        if not self.pattern_goals.get(pattern_id) or not self.pattern_goals[pattern_id]:
            print(f"Warning: Pattern {pattern_id} has no goals. Assigning a random free position.")
            all_free_for_goal = list(zip(*np.where(self.maze == 0)))
            if start_pos in all_free_for_goal and len(all_free_for_goal) > 1 : all_free_for_goal.remove(start_pos)
            if not all_free_for_goal: goal_pos = start_pos 
            else: goal_pos = random.choice(all_free_for_goal)
            if pattern_id not in self.pattern_goals: self.pattern_goals[pattern_id] = []
            self.pattern_goals[pattern_id].append(goal_pos) 
        else:
            goal_pos = random.choice(self.pattern_goals[pattern_id])


        self.current_state = MazeState(
            position=start_pos, target_pattern=self.patterns[pattern_id].copy(),
            target_position=goal_pos, visited_positions=[start_pos],
            steps_taken=0, max_steps=self.max_episode_steps)
        return self._get_observation(), self.current_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_state is None: self.reset()

        action_enum = Action(action)
        new_pos_candidate = self._get_new_position(self.current_state.position, action_enum)
        
        prev_pos = self.current_state.position 

        if self._is_valid_position(new_pos_candidate):
            self.current_state.position = new_pos_candidate
        
        if self.current_state.position not in self.current_state.visited_positions:
             self.current_state.visited_positions.append(self.current_state.position)
        elif len(self.current_state.visited_positions) == 0 : 
            self.current_state.visited_positions.append(self.current_state.position)


        self.current_state.steps_taken += 1
        reward = self._calculate_reward(action_enum, prev_pos) 

        done = (self.current_state.position == self.current_state.target_position or
                self.current_state.steps_taken >= self.current_state.max_steps)
        info = {'reached_goal': self.current_state.position == self.current_state.target_position,
                'steps_taken': self.current_state.steps_taken,
                'target_pattern_id': self._get_pattern_id(self.current_state.target_pattern)}
        return self._get_observation(), reward, done, info

    def _get_new_position(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        row, col = pos
        if action == Action.UP: return (row - 1, col)
        elif action == Action.DOWN: return (row + 1, col)
        elif action == Action.LEFT: return (row, col - 1)
        elif action == Action.RIGHT: return (row, col + 1)
        return pos

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        row, col = pos
        return (0 <= row < self.height and 0 <= col < self.width and self.maze[row, col] == 0)

    def _calculate_reward(self, action: Action, prev_pos: Tuple[int,int]) -> float:
        pos = self.current_state.position
        target_pos = self.current_state.target_position
        if pos == target_pos: return 10.0

        current_dist = euclidean(pos, target_pos)
        max_dist = euclidean((0,0), (self.height-1, self.width-1)) + 1e-9
        distance_reward = 1.0 - (current_dist / max_dist)
        time_penalty = -0.01
        
        revisit_penalty = 0.0
        if len(self.current_state.visited_positions) > 1 and \
           self.current_state.position in self.current_state.visited_positions[:-1]: 
            revisit_penalty = -0.05

        stay_penalty = -0.1 if action == Action.STAY and pos != target_pos else 0.0
        
        wall_hit_penalty = 0.0
        if action != Action.STAY and pos == prev_pos: 
            wall_hit_penalty = -0.2


        return distance_reward + time_penalty + revisit_penalty + stay_penalty + wall_hit_penalty

    def _get_observation(self) -> np.ndarray:
        if self.current_state is None: 
            pos_norm = np.zeros(2)
            local_field = np.ones(9) 
            distance_sensors = np.zeros(4)
            pattern_input = np.zeros(self.pattern_dim)
            steps_remaining = [0.0]
        else:
            pos = self.current_state.position
            target_pattern = self.current_state.target_pattern
            pos_norm = np.array([pos[0]/(self.height-1+1e-9), pos[1]/(self.width-1+1e-9)])
            local_field = self._get_local_field(pos)
            distance_sensors = self._get_distance_sensors(pos)
            pattern_input = target_pattern.copy() if target_pattern is not None else np.zeros(self.pattern_dim)
            steps_remaining = [(self.current_state.max_steps - self.current_state.steps_taken) /
                               (self.current_state.max_steps + 1e-9)]
        return np.concatenate([pos_norm, local_field, distance_sensors, pattern_input, steps_remaining]).astype(np.float32)

    def _get_local_field(self, pos: Tuple[int, int]) -> np.ndarray:
        row, col = pos
        local_field = np.ones(9, dtype=float)
        idx = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = row + dr, col + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    local_field[idx] = float(self.maze[r, c])
                idx +=1
        return local_field

    def _get_distance_sensors(self, pos: Tuple[int, int]) -> np.ndarray:
        row, col = pos
        distances = []
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        max_dim = max(self.height, self.width) + 1e-9
        for dr, dc in directions:
            distance = 0
            r_check, c_check = row+dr, col+dc
            while self._is_valid_position((r_check, c_check)):
                distance +=1; r_check += dr; c_check += dc
            distances.append(distance / max_dim)
        return np.array(distances)

    def _get_pattern_id(self, pattern: np.ndarray) -> int:
        if self.n_patterns == 0 or pattern is None: return -1
        similarities = np.array([np.dot(pattern, p_i) for p_i in self.patterns])
        return np.argmax(similarities) if similarities.size > 0 else -1

    def get_render_data(self, state_to_render: MazeState) -> Tuple[np.ndarray, str]:
        """Computes the visual representation and title for a given state."""
        maze_visual = self.maze.copy().astype(float)
        pos = state_to_render.position
        maze_visual[pos] = 0.3
        target_pos = state_to_render.target_position
        maze_visual[target_pos] = 0.7
        for visit_pos in state_to_render.visited_positions[:-1]:
            if visit_pos != pos and visit_pos != target_pos:
                maze_visual[visit_pos] = 0.5
        
        pattern_id_render = self._get_pattern_id(state_to_render.target_pattern)
        title_str = (f'Pattern {pattern_id_render}, Step: {state_to_render.steps_taken}/{state_to_render.max_steps}\n'
                     f'Pos: {pos}, Target: {target_pos}')
        return maze_visual, title_str

    def render(self, save_path: Optional[str] = None):
        if self.current_state is None:
            print("Cannot render, environment not reset.")
            return

        maze_visual, title_str = self.get_render_data(self.current_state)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(maze_visual, cmap='RdYlBu_r', vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, ticks=[0,0.3,0.5,0.7,1])
        cbar.ax.set_yticklabels(['Free','Current','Visited','Target','Wall'])
        ax.set_xticks(np.arange(self.width)); ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels(np.arange(self.width)); ax.set_yticklabels(np.arange(self.height))
        ax.grid(True,which='both',color='k',linestyle='-',linewidth=0.5,alpha=0.3)
        ax.set_xticks(np.arange(-.5,self.width,1),minor=True); ax.set_yticks(np.arange(-.5,self.height,1),minor=True)
        ax.grid(True,which='minor',color='k',linestyle='-',linewidth=1,alpha=0.4)
        ax.set_title(title_str)
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show(block=False); plt.pause(0.1)

class NavigationDataGenerator:
    def __init__(self, env: LabyrinthEnvironment, seed: int = 42):
        self.env = env
        np.random.seed(seed); random.seed(seed)

    def generate_pattern_recognition_data(self, n_samples:int=1000) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if self.env.n_patterns == 0: return np.array(X), np.array(y)
        for _ in range(n_samples):
            pattern_id = np.random.randint(0, self.env.n_patterns)
            pattern = self.env.patterns[pattern_id]
            noisy_pattern = pattern + np.random.normal(0,0.1,pattern.shape)
            target = np.zeros(self.env.n_patterns); target[pattern_id]=1
            X.append(noisy_pattern); y.append(target)
        return np.array(X), np.array(y)

    def generate_spatial_localization_data(self, n_samples:int=1000) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        free_positions = list(zip(*np.where(self.env.maze == 0)))
        if not free_positions: return np.array(X), np.array(y)
        original_state = self.env.current_state
        for _ in range(n_samples):
            pos = random.choice(free_positions)
            dummy_pattern = self.env.patterns[0] if self.env.n_patterns > 0 else np.zeros(self.env.pattern_dim)
            self.env.current_state = MazeState(pos, dummy_pattern, pos, [pos], 0, 100)
            local_field = self.env._get_local_field(pos)
            distance_sensors = self.env._get_distance_sensors(pos)
            sensory_input = np.concatenate([local_field, distance_sensors])
            target_pos = np.array([pos[0]/(self.env.height-1+1e-9), pos[1]/(self.env.width-1+1e-9)])
            X.append(sensory_input); y.append(target_pos)
        self.env.current_state = original_state
        return np.array(X), np.array(y)

    def generate_direction_prediction_data(self, n_samples:int=1000) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if self.env.n_patterns == 0: return np.array(X), np.array(y)
        free_positions = list(zip(*np.where(self.env.maze == 0)))
        if not free_positions: return np.array(X), np.array(y)

        for _ in range(n_samples):
            pattern_id = np.random.randint(0, self.env.n_patterns)
            obs, state = self.env.reset(pattern_id)
            state.position = random.choice(free_positions)
            self.env.current_state = state
            obs = self.env._get_observation()
            
            target_pos = state.target_position; curr_pos = state.position
            dr = target_pos[0]-curr_pos[0]; dc = target_pos[1]-curr_pos[1]
            opt_action_val = Action.STAY.value
            if abs(dr) > abs(dc): opt_action_val = Action.DOWN.value if dr > 0 else Action.UP.value
            elif abs(dc) > 0: opt_action_val = Action.RIGHT.value if dc > 0 else Action.LEFT.value
            action_target = np.zeros(len(Action)); action_target[opt_action_val]=1
            X.append(obs); y.append(action_target)
        return np.array(X), np.array(y)

    def generate_working_memory_data(self, n_samples:int=500) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if self.env.n_patterns == 0: return np.array(X), np.array(y)
        for _ in range(n_samples):
            pattern_id = np.random.randint(0, self.env.n_patterns)
            obs, state = self.env.reset(pattern_id)
            seq_len = np.random.randint(3,8)
            for _ in range(seq_len):
                action = np.random.randint(0,len(Action))
                obs,_,done,_ = self.env.step(action)
                if done: break
            current_pos = self.env.current_state.position
            was_visited = current_pos in self.env.current_state.visited_positions[:-1]
            target = np.array([1.0 if was_visited else 0.0])
            X.append(obs); y.append(target)
        return np.array(X), np.array(y)

VALUE_TARGET_MIN = -10.0 
VALUE_TARGET_MAX = 30.0 
VALUE_TARGET_RANGE = VALUE_TARGET_MAX - VALUE_TARGET_MIN

class ImprovedNavigationDataGenerator(NavigationDataGenerator):
    def generate_integrated_navigation_data_improved(self, n_episodes: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y_reg, y_clf = [], [], []
        if self.env.n_patterns == 0 and n_episodes > 0: 
            print("Warning: No patterns in env for integrated data generation.")
            dummy_obs_len = len(self.env._get_observation()) if self.env.current_state else 2+9+4+self.env.pattern_dim+1
            return np.empty((0, dummy_obs_len)), np.empty((0,3)), np.empty((0, len(Action)))


        for episode in range(n_episodes):
            if episode % 100 == 0: print(f"Generating integrated data episode {episode}/{n_episodes}")
            difficulty = min(1.0, episode / (n_episodes*0.7 if n_episodes > 0 else 1.0))
            current_max_steps = int(20 + difficulty * (self.env.max_episode_steps - 20))
            current_max_steps = max(10, current_max_steps)

            obs, state = self.env.reset() 
            state.max_steps = current_max_steps
            self.env.current_state.max_steps = current_max_steps 

            episode_data = []
            for step_int in range(state.max_steps):
                current_obs = obs.copy()
                epsilon = 0.8 * (1.0-(episode/(n_episodes if n_episodes > 0 else 1.0))) + 0.1
                action_to_take = self._get_optimal_action(state.position, state.target_position) \
                                 if np.random.random() > epsilon else np.random.randint(0,len(Action))
                
                obs, reward, done, info = self.env.step(action_to_take)
                state = self.env.current_state 

                enhanced_reward = self._calculate_enhanced_reward(state, Action(action_to_take), reward, done, info)
                future_val_est = 0.0
                if not done:
                    steps_to_goal_h = self._manhattan_distance(state.position, state.target_position)
                    future_val_est = max(0, (state.max_steps - steps_to_goal_h) / (state.max_steps + 1e-9))
                elif info['reached_goal']: future_val_est = 1.0
                
                value_target = enhanced_reward + 0.9 * future_val_est
                value_target_norm = np.clip((value_target - VALUE_TARGET_MIN) / (VALUE_TARGET_RANGE + 1e-9), 0.0, 1.0)
                
                pos_target = np.array([state.position[0]/(self.env.height-1+1e-9), state.position[1]/(self.env.width-1+1e-9)])
                reg_target = np.array([value_target_norm, pos_target[0], pos_target[1]])
                
                opt_action_clf = self._get_optimal_action(state.position, state.target_position)
                clf_target = np.zeros(len(Action)); clf_target[opt_action_clf] = 1.0
                episode_data.append((current_obs, reg_target, clf_target))
                if done:
                    if info['reached_goal']:
                        for _ in range(3): episode_data.append((current_obs, reg_target, clf_target))
                    break
            for obs_i, reg_i, clf_i in episode_data:
                X.append(obs_i); y_reg.append(reg_i); y_clf.append(clf_i)
        
        if not X: 
            dummy_obs_len = len(self.env._get_observation()) if self.env.current_state else 2+9+4+self.env.pattern_dim+1
            return np.empty((0, dummy_obs_len)), np.empty((0,3)), np.empty((0, len(Action)))
        return np.array(X), np.array(y_reg), np.array(y_clf)

    def _get_optimal_action(self, current_pos:Tuple[int,int], target_pos:Tuple[int,int]) -> int:
        dr=target_pos[0]-current_pos[0]; dc=target_pos[1]-current_pos[1]
        if abs(dr)>abs(dc): return Action.DOWN.value if dr > 0 else Action.UP.value
        elif abs(dc)>0: return Action.RIGHT.value if dc > 0 else Action.LEFT.value
        return Action.STAY.value

    def _calculate_enhanced_reward(self, state:MazeState, action_enum:Action, base_reward:float, done:bool, info:Dict) -> float:
        reward = base_reward
        if info['reached_goal']: reward += 20.0
        
        prev_pos = state.visited_positions[-2] if len(state.visited_positions) >= 2 else state.position
        prev_dist_m = self._manhattan_distance(prev_pos, state.target_position)
        current_dist_m = self._manhattan_distance(state.position, state.target_position)

        if current_dist_m < prev_dist_m: reward += 0.5
        elif current_dist_m > prev_dist_m and state.position != prev_pos: reward -= 0.2
        if action_enum == Action.STAY and not info['reached_goal']: reward -= 0.15
        if state.steps_taken > 0 and state.position == prev_pos and action_enum != Action.STAY: reward -= 0.3 
        if state.position not in state.visited_positions[:-1] and self.env.maze[state.position]==0: reward += 0.05
        return reward

    def _manhattan_distance(self, pos1:Tuple[int,int], pos2:Tuple[int,int]) -> int:
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

class CurriculumNavigationTrainer:
    def __init__(self, env:LabyrinthEnvironment, network_params:Dict[str,Any], seed:int=42):
        self.env = env
        self.data_generator = ImprovedNavigationDataGenerator(env, seed)
        self.network_params = network_params; self.networks = {}

    def train_subtask_1_pattern_recognition(self, n_samples:int=2000, n_epochs:int=500) -> Optional[MultiTaskComplexLearner]:
        print("\n🎯 Training Sub-task 1: Pattern Recognition"); print("="*50)
        X,y_clf = self.data_generator.generate_pattern_recognition_data(n_samples)
        if X.shape[0]==0: print("No data for pattern recognition. Skipping."); return None
        y_reg = np.zeros((X.shape[0],1))
        learner = MultiTaskComplexLearner(X.shape[1],1,self.env.n_patterns,**self.network_params)
        learner.learn(X,y_reg,y_clf,n_epochs,batch_size=32,validation_split=0.2,verbose=True,min_epochs_no_improve=50,patience_no_improve=100)
        self.networks['pattern_recognition'] = learner; return learner

    def train_subtask_2_spatial_localization(self, n_samples:int=2000, n_epochs:int=800) -> Optional[MultiTaskComplexLearner]:
        print("\n🗺️ Training Sub-task 2: Spatial Localization"); print("="*50)
        X,y_reg = self.data_generator.generate_spatial_localization_data(n_samples)
        if X.shape[0]==0: print("No data for spatial localization. Skipping."); return None
        y_clf = np.zeros((X.shape[0],1))
        learner = MultiTaskComplexLearner(X.shape[1],2,1,**self.network_params) 
        learner.learn(X,y_reg,y_clf,n_epochs,batch_size=32,validation_split=0.2,verbose=True,min_epochs_no_improve=50,patience_no_improve=100)
        self.networks['spatial_localization'] = learner; return learner

    def train_subtask_3_direction_prediction(self, n_samples:int=3000, n_epochs:int=1000) -> Optional[MultiTaskComplexLearner]:
        print("\n🧭 Training Sub-task 3: Direction Prediction"); print("="*50)
        X,y_clf = self.data_generator.generate_direction_prediction_data(n_samples)
        if X.shape[0]==0: print("No data for direction prediction. Skipping."); return None
        y_reg = np.zeros((X.shape[0],1))
        learner = MultiTaskComplexLearner(X.shape[1],1,len(Action),**self.network_params)
        learner.learn(X,y_reg,y_clf,n_epochs,batch_size=32,validation_split=0.2,verbose=True,min_epochs_no_improve=50,patience_no_improve=100)
        self.networks['direction_prediction'] = learner; return learner

    def train_subtask_4_working_memory(self, n_samples:int=1500, n_epochs:int=600) -> Optional[MultiTaskComplexLearner]:
        print("\n🧠 Training Sub-task 4: Working Memory"); print("="*50)
        X,y_clf = self.data_generator.generate_working_memory_data(n_samples)
        if X.shape[0]==0: print("No data for working memory. Skipping."); return None
        y_reg=np.zeros((X.shape[0],1)); y_clf_binary=y_clf.reshape(-1,1)
        learner = MultiTaskComplexLearner(X.shape[1],1,1,**self.network_params)
        learner.learn(X,y_reg,y_clf_binary,n_epochs,batch_size=32,validation_split=0.2,verbose=True,min_epochs_no_improve=50,patience_no_improve=100)
        self.networks['working_memory'] = learner; return learner

class ImprovedNavigationAgent:
    def __init__(self, learner:MultiTaskComplexLearner, exploration_rate:float=0.1):
        self.learner=learner; self.exploration_rate=exploration_rate
    def select_action(self, obs:np.ndarray, greedy:bool=False) -> int:
        if not self.learner: return np.random.randint(0,len(Action))
        obs_reshaped = obs.reshape(1,-1)
        _,pred_clf_probs,_ = self.learner.predict(obs_reshaped)
        if not greedy and np.random.random() < self.exploration_rate:
            return np.random.randint(0,len(Action))
        if pred_clf_probs.shape[1]==0: return np.random.randint(0,len(Action))
        action_probs_sample = pred_clf_probs[0] + np.random.normal(0,0.001,pred_clf_probs.shape[1])
        return np.argmax(action_probs_sample)

def evaluate_navigation_performance(learner:Optional[MultiTaskComplexLearner], env:LabyrinthEnvironment, n_test_episodes:int=50) -> Dict[str,float]:
    print(f"\n🔬 Evaluating Navigation Performance ({n_test_episodes} episodes)")
    if learner is None:
        print("Learner is None, cannot evaluate.")
        return {'success_rate':0.0,'average_steps':float(env.max_episode_steps),'average_reward':0.0,
                'efficiency':0.0,'total_goal_reaches':0,'total_steps_taken':n_test_episodes*env.max_episode_steps}
    results={'success_rate':0.0,'average_steps':0.0,'average_reward':0.0,'efficiency':0.0,'total_goal_reaches':0,'total_steps_taken':0}
    successful_episodes=0; total_steps_all_episodes=0; total_reward_all_episodes=0.0
    agent = ImprovedNavigationAgent(learner, exploration_rate=0.0)
    for _ in range(n_test_episodes):
        obs,state = env.reset(); episode_reward_sum=0.0; episode_steps=0
        for _ in range(state.max_steps):
            action = agent.select_action(obs,greedy=True)
            obs,reward,done,info = env.step(action)
            episode_reward_sum += reward; episode_steps = state.steps_taken
            if done:
                if info['reached_goal']: successful_episodes+=1
                break
        total_steps_all_episodes+=episode_steps; total_reward_all_episodes+=episode_reward_sum
    
    results['success_rate'] = successful_episodes/n_test_episodes if n_test_episodes>0 else 0.0
    results['average_steps'] = total_steps_all_episodes/n_test_episodes if n_test_episodes>0 else 0.0
    results['average_reward'] = total_reward_all_episodes/n_test_episodes if n_test_episodes>0 else 0.0
    results['efficiency'] = successful_episodes/(total_steps_all_episodes+1e-9)
    results['total_goal_reaches']=successful_episodes; results['total_steps_taken']=total_steps_all_episodes
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average Steps: {results['average_steps']:.1f}")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Efficiency: {results['efficiency']:.4f}")
    return results

def visualize_curriculum_learning_results(trainer:CurriculumNavigationTrainer, env:LabyrinthEnvironment, final_integrated_learner:Optional[MultiTaskComplexLearner]=None):
    n_subtasks = len(trainer.networks)
    n_total_plots = n_subtasks + 2 
    ncols = 3 if n_total_plots > 4 else 2
    nrows = (n_total_plots + ncols - 1)//ncols
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*5))
    fig.suptitle('🐭 Rat Navigation - Curriculum Learning Results',fontsize=16,y=0.98)
    axes_flat = axes.flatten(); plot_idx=0
    subtasks_to_plot = ['pattern_recognition','spatial_localization','direction_prediction','working_memory']
    if 'integrated_navigation' in trainer.networks and trainer.networks['integrated_navigation']==final_integrated_learner:
        if 'integrated_navigation' not in subtasks_to_plot: subtasks_to_plot.append('integrated_navigation')

    for task_name in subtasks_to_plot:
        if task_name in trainer.networks and trainer.networks[task_name] is not None:
            if plot_idx < len(axes_flat):
                ax=axes_flat[plot_idx]; network=trainer.networks[task_name]
                epochs=network.training_history['epoch']
                if epochs:
                    ax.plot(epochs,network.training_history['loss'],label='Train Loss',lw=2)
                    if network.training_history.get('val_loss') and any(network.training_history['val_loss']):
                        ax.plot(epochs,network.training_history['val_loss'],label='Val Loss',lw=2,ls='--')
                ax.set_title(f'Task: {task_name.replace("_"," ").title()}'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
                ax.legend(); ax.grid(True,alpha=0.3); plot_idx+=1
    
    if plot_idx < len(axes_flat):
        ax_maze=axes_flat[plot_idx]
        ax_maze.imshow(env.maze,cmap='binary_r'); ax_maze.set_title('Maze Layout\n(White=Free, Black=Wall)')
        ax_maze.set_xlabel('X Pos'); ax_maze.set_ylabel('Y Pos')
        if env.n_patterns > 0:
            # Use a consistent colormap for pattern goals
            # cmap = plt.cm.get_cmap('tab10', env.n_patterns) # 'tab10' has distinct colors
            # Using fixed list for 3 patterns for better distinction if tab10 is not good enough
            goal_colors = ['purple', 'cyan', 'yellow', 'green', 'orange']


            for p_id,goals in env.pattern_goals.items():
                if goals: ax_maze.scatter(goals[0][1],goals[0][0],color=goal_colors[p_id % len(goal_colors)],s=100,marker='*',label=f'P{p_id} Goal')
            ax_maze.legend(fontsize='small',loc='best')
        plot_idx+=1

    if plot_idx < len(axes_flat):
        ax_nav=axes_flat[plot_idx]
        if final_integrated_learner:
            obs_viz,state_viz = env.reset(pattern_id=0 if env.n_patterns>0 else None)
            agent_viz = ImprovedNavigationAgent(final_integrated_learner,exploration_rate=0.0)
            path=[state_viz.position]; max_viz_steps=env.max_episode_steps
            for _ in range(max_viz_steps):
                action=agent_viz.select_action(obs_viz,greedy=True)
                obs_viz,_,done,info = env.step(action)
                state_viz=env.current_state; path.append(state_viz.position)
                if done: break
            maze_viz_nav = env.maze.copy().astype(float)
            for i,p in enumerate(path): maze_viz_nav[p] = 0.3+0.4*(i/max(1,len(path)-1))
            if path: maze_viz_nav[path[0]]=0.8 
            maze_viz_nav[state_viz.target_position]=1.0
            ax_nav.imshow(maze_viz_nav,cmap='RdYlBu_r',vmin=0,vmax=1)
            p_id_title = env._get_pattern_id(state_viz.target_pattern)
            ax_nav.set_title(f'Example Path (Ptn {p_id_title})\n(Darker=Later)')
            ax_nav.set_xlabel('X Pos'); ax_nav.set_ylabel('Y Pos')
        else: ax_nav.text(0.5,0.5,"Integrated Learner\nNot Available",ha='center',va='center',transform=ax_nav.transAxes)
        plot_idx+=1
    for i in range(plot_idx,len(axes_flat)): fig.delaxes(axes_flat[i])
    plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show(); plt.pause(0.1)


def animate_navigation(env: LabyrinthEnvironment, frames_data: List[Tuple[np.ndarray, str]], interval: int = 300):
    """Creates and shows a navigation animation."""
    if not frames_data:
        print("No frames to animate.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) # Slightly larger for bigger maze
    initial_maze_visual, initial_title = frames_data[0]

    im = ax.imshow(initial_maze_visual, cmap='RdYlBu_r', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.3, 0.5, 0.7, 1])
    cbar.ax.set_yticklabels(['Free', 'Current', 'Visited', 'Target', 'Wall'])

    ax.set_xticks(np.arange(env.width)); ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels(np.arange(env.width)); ax.set_yticklabels(np.arange(env.height))
    ax.grid(True, which='both', color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, alpha=0.4)
    
    title_obj = ax.set_title(initial_title)

    def update(frame_idx: int):
        maze_visual, title_text = frames_data[frame_idx]
        im.set_data(maze_visual)
        title_obj.set_text(title_text)
        return [im, title_obj]

    anim = FuncAnimation(fig, update, frames=len(frames_data),
                         interval=interval, blit=True, repeat=False)
    plt.show()
    # try:
    #     print("Saving animation... (this may take a moment)")
    #     anim.save('harder_navigation_animation.gif', writer='pillow', fps=1000/interval)
    #     print("Animation saved as harder_navigation_animation.gif")
    # except Exception as e:
    #     print(f"Could not save animation: {e}. Is pillow installed?")


# Main execution block
if __name__ == "__main__":
    print("🐭 RAT LABYRINTH NAVIGATION - Brain-Inspired Multi-Task Learning")
    print("=" * 80)
    env_seed = 42
    
    # --- Create HARDER environment ---
    env = LabyrinthEnvironment(
        maze_size=(15, 15),       # Increased size
        wall_density=0.30,      # Increased density
        n_patterns=3,           # Kept same for now
        pattern_dim=6,          # Kept same
        max_episode_steps=150,  # Increased steps
        seed=env_seed
    )
    print(f"Created HARDER maze environment: {env.maze_size}, Wall Density: {env.maze.sum() / env.maze.size:.2f}")
    print(f"Patterns: {env.n_patterns}, Pattern dim: {env.pattern_dim}, Max Steps: {env.max_episode_steps}")

    network_params = {'n_hidden_circuits':40, # Slightly increased capacity
                      'n_internal_units_per_circuit':12, # Slightly increased
                      'learning_rate':0.0003, # Potentially smaller LR for harder task
                      'loss_weight_regression':1.0, 
                      'loss_weight_classification':1.0, 
                      'microcircuit_aggregation':'max', 
                      'l1_activation_lambda':0.000005, # Reduced L1
                      'use_kwta_on_circuits':True, 
                      'kwta_k_circuits':20, # k for kWTA
                      'seed':env_seed}
    
    trainer = CurriculumNavigationTrainer(env, network_params, seed=env_seed)

    print("\n🎓 Starting Curriculum Learning (on harder maze)...")
    # Sub-task training might also need more samples/epochs for a harder env,
    # but let's keep them moderate for now to see the impact.
    trainer.train_subtask_1_pattern_recognition(n_samples=1200, n_epochs=350)
    trainer.train_subtask_2_spatial_localization(n_samples=1200, n_epochs=350)
    trainer.train_subtask_3_direction_prediction(n_samples=1800, n_epochs=600)
    trainer.train_subtask_4_working_memory(n_samples=1000, n_epochs=350)

    print("\n🚀 Phase 2: Training Integrated Navigation Model (on harder maze)"); print("="*50)
    # Generate more data for the harder maze
    X_integrated, y_reg_integrated, y_clf_integrated = \
        trainer.data_generator.generate_integrated_navigation_data_improved(n_episodes=1000) # Increased episodes
    
    final_integrated_learner = None
    if X_integrated.shape[0] > 0:
        final_integrated_learner = MultiTaskComplexLearner(
            X_integrated.shape[1], y_reg_integrated.shape[1], y_clf_integrated.shape[1], **network_params)
        print(f"Integrated Model Input Features: {X_integrated.shape[1]}")
        print(f"Integrated Model Regression Outputs: {y_reg_integrated.shape[1]}")
        print(f"Integrated Model Classification Tasks: {y_clf_integrated.shape[1]}")
        # Train longer for the harder maze
        final_integrated_learner.learn(X_integrated,y_reg_integrated,y_clf_integrated,n_epochs=1500,batch_size=64,
                                     validation_split=0.15,min_epochs_no_improve=100,patience_no_improve=200,verbose=True)
        trainer.networks['integrated_navigation'] = final_integrated_learner
    else: print("No data for integrated navigation. Skipping training.")

    visualize_curriculum_learning_results(trainer, env, final_integrated_learner=final_integrated_learner)

    if final_integrated_learner:
        print("\n🧪 Evaluating Final Integrated Navigation Performance (on harder maze)...")
        results = evaluate_navigation_performance(final_integrated_learner, env, n_test_episodes=50)
        
        print("\n🎬 Demonstrating Navigation with Trained Agent (Animation on harder maze)...")
        obs_demo, state_demo = env.reset(pattern_id=0 if env.n_patterns > 0 else None)
        
        animation_frames_data = [env.get_render_data(state_demo)] 
        
        agent_demo = ImprovedNavigationAgent(final_integrated_learner, exploration_rate=0.0)
        # Use state_demo.max_steps which is now 150 for the harder maze
        for step_demo in range(state_demo.max_steps): 
            action_demo = agent_demo.select_action(obs_demo, greedy=True)
            action_name_demo = Action(action_demo).name
            print(f"Step {step_demo+1}/{state_demo.max_steps}: Action = {action_name_demo}, Pos = {env.current_state.position}")
            obs_demo, reward_demo, done_demo, info_demo = env.step(action_demo)
            animation_frames_data.append(env.get_render_data(env.current_state))
            if done_demo:
                print(f"Episode finished! Goal reached: {info_demo['reached_goal']} in {env.current_state.steps_taken} steps.")
                break
        
        if animation_frames_data:
            animate_navigation(env, animation_frames_data, interval=250) # Slightly faster interval for longer episodes
        else:
            print("No frames collected for animation.")

    else: print("Skipping final evaluation and demo as integrated learner was not trained.")
    print("\n✅ Rat Navigation Curriculum Learning Completed!")
