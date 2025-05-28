import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import os  # for caching generated dataset

# Brian2 Simulation Parameters
b2.defaultclock.dt = 0.1 * b2.ms  # Simulation time step
DURATION = 200 * b2.ms          # Duration of each simulation run
N_SAMPLES = 10000                # Number of data samples to generate
SEQUENCE_LENGTH = int(DURATION / b2.defaultclock.dt) # Timesteps per sample

tau_rec = 200 * b2.ms
tau_facil = 50 * b2.ms
U = 0.4
g_max = 1.0 # Make sure g_max is defined, e.g., at the global scope or passed appropriately

# Brian2 Model
# We'll model the conductance directly for simplicity in matching with CNN output
# x: available resources, u: utilization factor
# On spike: x decreases by u*x, u increases, conductance pulse g_max*u*x
eqs_synapse = """
    dx/dt = (1-x)/tau_rec : 1 (event-driven)
    du/dt = -u/tau_facil : 1 (event-driven)
    g_syn : 1 # Synaptic conductance
"""

# Presynaptic neuron (Poisson spike generator)
# We will manually set spike times for more control in data generation
# For the CNN, we'll feed a binary spike train

def generate_synaptic_data(n_samples, seq_len, actual_dt):
    """Generates input spike trains and output synaptic conductances."""
    all_input_spikes = np.zeros((n_samples, seq_len))
    all_output_conductances = np.zeros((n_samples, seq_len))

    for i in range(n_samples):
        if i % (n_samples // 10) == 0:
            print(f"Generating sample {i+1}/{n_samples}")

        rate = np.random.uniform(5, 50) * b2.Hz
        input_spike_train = np.random.rand(seq_len) < (rate * actual_dt)

        # Get the time step indices where spikes occur
        time_step_indices_of_spikes = np.where(input_spike_train)[0]
        # Calculate the actual times of these spikes
        actual_spike_times = time_step_indices_of_spikes * actual_dt

        # For SpikeGeneratorGroup with N=1, the neuron_indices_for_spikes must all be 0
        # The number of spikes is len(actual_spike_times)
        num_spikes = len(actual_spike_times)
        neuron_indices_for_spikes = np.zeros(num_spikes, dtype=int)

        G = b2.NeuronGroup(1, 'v:1', threshold='v>0.9', reset='v=0') # Dummy target neuron

        # Correctly create SpikeGeneratorGroup
        # If there are no spikes, actual_spike_times will be empty,
        # and neuron_indices_for_spikes will also be empty, which is fine.
        spike_gen = b2.SpikeGeneratorGroup(1, neuron_indices_for_spikes, actual_spike_times)

        S = b2.Synapses(
            spike_gen, G,
            model="""
                dx/dt = (1-x)/tau_rec : 1 (event-driven)
                du/dt = -u/tau_facil : 1 (event-driven)
                dg_total/dt = -g_total / (10*ms) : 1 (clock-driven) # Explicitly clock-driven
                g_pulse : 1 # Stores the magnitude of the last pulse (can be monitored)
            """,
            on_pre="""
                g_pulse = g_max * u * x  # Calculate the pulse for this event
                g_total += g_pulse       # CORRECT: Add pulse to the synaptic variable g_total
                x = x - u * x            # Depress: reduce available resources
                u = u + U * (1 - u)      # Facilitate: increase utilization
            """,
            method='exact'
        )
        S.connect(i=0, j=0)
        S.x = 1.0  # Initial condition: full resources
        S.u = U    # Initial condition: baseline utilization
        S.g_total = 0.0 # Initialize synaptic conductance
        S.g_pulse = 0.0 # Initialize g_pulse if you want a defined start value
        state_mon = b2.StateMonitor(S, ['g_total', 'x', 'u', 'g_pulse'], record=True, dt=actual_dt)

        net = b2.Network(spike_gen, G, S, state_mon)
        net.run(DURATION, report=None)  # report=None can speed it up slightly for many runs

        all_input_spikes[i, :] = input_spike_train.astype(float)
        recorded_g = state_mon.g_total[0]
        if len(recorded_g) < seq_len:
            padding = np.zeros(seq_len - len(recorded_g))
            all_output_conductances[i, :] = np.concatenate((recorded_g, padding))
        else:
            all_output_conductances[i, :] = recorded_g[:seq_len]

    return all_input_spikes, all_output_conductances

# Cache Brian2-generated data to avoid regeneration every run
data_file = 'synapse_data.npz'
actual_dt_for_generation = b2.defaultclock.dt  # Brian2 dt with units
if os.path.exists(data_file):
    print(f"Loading cached synapse data from {data_file}...")
    data = np.load(data_file)
    input_spikes, output_conductances = data['X'], data['y']
else:
    print("Generating training data with Brian2...")
    print(f"Using actual_dt_for_generation: {actual_dt_for_generation}")
    input_spikes, output_conductances = generate_synaptic_data(
        N_SAMPLES, SEQUENCE_LENGTH, actual_dt_for_generation
    )
    print(f"Saving generated data to {data_file}...")
    np.savez(data_file, X=input_spikes, y=output_conductances)
print(f"Input spikes shape: {input_spikes.shape}")
print(f"Output conductances shape: {output_conductances.shape}")

# Reshape for CNN (add channel dimension for Conv1D)
X = input_spikes.reshape((N_SAMPLES, SEQUENCE_LENGTH, 1))
y = output_conductances.reshape((N_SAMPLES, SEQUENCE_LENGTH, 1)) # Predicting the whole sequence

# Normalize output conductances (important for training neural networks)
# We'll use a scaler for y. For X, it's already 0 or 1.
# Fit scaler on training data only, or on all data if not strictly separating test set yet
# For simplicity here, fit on all generated y. In a real scenario, fit on train set.
scaler_y = StandardScaler() # Or MinMaxScaler(feature_range=(0,1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Plot a sample to verify data generation
sample_idx = 0
time_axis = np.arange(SEQUENCE_LENGTH) * (b2.defaultclock.dt / b2.ms)

fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:red'
ax1.set_xlabel(f'Time (ms), dt={b2.defaultclock.dt / b2.ms:.2f} ms')
ax1.set_ylabel('Input Spikes', color=color)
# Plot spikes as stems
spike_times_plot = np.where(X[sample_idx, :, 0] > 0.5)[0] * (b2.defaultclock.dt / b2.ms)
if spike_times_plot.size > 0:
    markerline, stemlines, baseline = ax1.stem(
        spike_times_plot,
        np.ones_like(spike_times_plot),
        linefmt='-', markerfmt='o', basefmt=" "
    )
    plt.setp(markerline, color=color)
    plt.setp(stemlines, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.5)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Synaptic Conductance (scaled)', color=color)
ax2.plot(time_axis, y_scaled[sample_idx, :, 0], color=color, alpha=0.7, label="Brian2 Output (Scaled)")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f"Sample Brian2 Synapse Input/Output (Sample {sample_idx})")
plt.show()


# Define callbacks for training
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# --- Deep Feedforward Network (MLP) ---
# Prepare input shape for DNN: flatten sequence dimension
input_shape = (SEQUENCE_LENGTH, 1)

def build_dnn_model(input_shape, hidden_layers=[512,256,128], dropout=0.5, l2_reg=1e-4):
    model = keras.Sequential(name="DNN_" + "x".join(map(str, hidden_layers)))
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for units in hidden_layers:
        model.add(layers.Dense(
            units,
            activation='elu',  # smoother activation
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    # final output to full sequence length
    seq_len = input_shape[0]
    model.add(layers.Dense(seq_len, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.0003),
        loss=keras.losses.Huber(),
    )
    model.summary()
    return model

# Train and evaluate DNN models
results = {}
for layers_cfg in [
# [1024,512,256,128,64],
]:
     key = f"DNN_{'_'.join(map(str,layers_cfg))}"
     print(f"\nTraining {key}")
     model = build_dnn_model(input_shape, hidden_layers=layers_cfg)
     history = model.fit(
         X_train, y_train.reshape(-1, X_train.shape[1]),
         epochs=500, batch_size=32,  # increased epochs for deeper training
         validation_data=(X_test, y_test.reshape(-1, X_test.shape[1])),
         callbacks=[early_stopping, reduce_lr],
         verbose=1
     )
     loss = model.evaluate(X_test, y_test.reshape(-1, X_test.shape[1]), verbose=0)
     results[key] = {'loss': loss, 'model': model, 'history': history.history}
     print(f"Test MSE for {key}: {loss:.6f}")

# Select best DNN
best_model_key = min(results, key=lambda k: results[k]['loss'])
best_model = results[best_model_key]['model']
best_loss = results[best_model_key]['loss']
print(f"\nBest model: {best_model_key} with MSE: {best_loss:.6f}")

# Visualize predictions of best DNN
n_vis_samples = 3
for i in range(n_vis_samples):
    sample_idx = np.random.randint(0, X_test.shape[0])
    input_sample = X_test[sample_idx:sample_idx+1]
    true_scaled = y_test[sample_idx]
    pred_scaled = best_model.predict(input_sample)[0]
    true_orig = scaler_y.inverse_transform(true_scaled.reshape(-1,1))
    pred_orig = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))
    time_axis = np.arange(SEQUENCE_LENGTH) * (b2.defaultclock.dt / b2.ms)
    fig, ax1 = plt.subplots(figsize=(15,7))
    color='tab:red'
    ax1.set_xlabel(f'Time (ms), dt={b2.defaultclock.dt/b2.ms:.2f} ms')
    ax1.set_ylabel('Input Spikes', color=color)
    spikes = np.where(input_sample[0,:,0]>0.5)[0]*(b2.defaultclock.dt/b2.ms)
    if spikes.size>0:
        marker, stems, base = ax1.stem(spikes, np.ones_like(spikes), linefmt='-', markerfmt='o', basefmt=' ')
        plt.setp(marker, color=color); plt.setp(stems, color=color)
    ax1.tick_params(axis='y', labelcolor=color); ax1.set_ylim(-0.1,1.5)
    ax2 = ax1.twinx(); col2='tab:blue'
    ax2.set_ylabel('Conductance', color=col2)
    ax2.plot(time_axis, true_orig[:,0], color=col2); ax2.plot(time_axis, pred_orig[:,0], color='tab:green')
    plt.title(f"Prediction vs True ({best_model_key}) Sample {sample_idx}")
    fig.tight_layout(); plt.show()

# After selecting the best model, export its weights to JSON for Rust integration
# Export model weights
best_weights = {}
for idx, layer in enumerate(best_model.layers):
    if hasattr(layer, 'kernel'):
        w, b = layer.get_weights()
        best_weights[f"layer_{idx}_kernel"] = w.tolist()
        best_weights[f"layer_{idx}_bias"] = b.tolist()

def save_weights_to_json(weights, path):
    with open(path, 'w') as f:
        json.dump(weights, f)
    print(f"Saved DNN weights to {path}")

save_weights_to_json(best_weights, 'dnn_weights.json')

# === Spike Function Approximation Section ===
# Use existing keras and layers imports; set K alias
K = keras.backend

# Define target rectangular spike for training approximators
def spike_target(x, width=0.005, height=1.0):
    y = np.zeros_like(x)
    y[np.abs(x) <= width] = height
    return y

# 1) Difference-of-sigmoids model
def build_sigmoid_diff_model(k=100.0, b=0.005, A=1.0):
    def diff_sigmoid(x):
        return A*(1/(1+K.exp(-k*(x+b))) - 1/(1+K.exp(-k*(x-b))))
    m = keras.Sequential([
        layers.InputLayer(input_shape=(1,)),
        layers.Lambda(diff_sigmoid)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

# 2) ReLU-based MLP
def build_relu_mlp():
    m = keras.Sequential([
        layers.InputLayer(input_shape=(1,)),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal', use_bias=False),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal', use_bias=False),
        layers.Dense(1, activation='linear')
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

# 3) Custom Gaussian activation
def build_custom_gaussian(sigma=0.005, mu=0.0, A=1.0):
    def gauss(x): return A*K.exp(-K.square(x-mu)/(2*K.square(sigma)))
    m = keras.Sequential([
        layers.InputLayer(input_shape=(1,)),
        layers.Lambda(gauss)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

# 4) Simple RNN transient model
def build_rnn_model():
    # Sequence-to-sequence RNN for synaptic conductance prediction
    m = keras.Sequential([
        layers.InputLayer(input_shape=(SEQUENCE_LENGTH,1)),
        layers.SimpleRNN(32, activation='tanh', return_sequences=True),
        layers.TimeDistributed(layers.Dense(1, activation='linear'))
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

# --- Synapse Dataset Approximation with Multiple Architectures ---
approximator_builders = {
    'ReLU_MLP_Seq': lambda: _build_seq_relu_mlp(),
}
def _build_seq_diff_sigmoid_model(k=100.0, b=0.005, A=1.0):
    inp = layers.Input(shape=(SEQUENCE_LENGTH,1))
    out = layers.Lambda(lambda x: A*(1/(1+K.exp(-k*(x+b))) - 1/(1+K.exp(-k*(x-b)))))(inp)
    m = keras.Model(inputs=inp, outputs=out)
    m.compile(optimizer='adam', loss='mse')
    return m
def _build_seq_custom_gaussian(sigma=0.005, mu=0.0, A=1.0):
    inp = layers.Input(shape=(SEQUENCE_LENGTH,1))
    out = layers.Lambda(lambda x: A*K.exp(-K.square(x-mu)/(2*K.square(sigma))))(inp)
    m = keras.Model(inputs=inp, outputs=out)
    m.compile(optimizer='adam', loss='mse')
    return m
def _build_seq_relu_mlp():
    m = keras.Sequential(name='ReLU_MLP_Seq')
    m.add(layers.Input(shape=(SEQUENCE_LENGTH,1)))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.Dense(SEQUENCE_LENGTH, activation='linear'))
    m.add(layers.Reshape((SEQUENCE_LENGTH,1)))
    m.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.00003), loss=keras.losses.Huber())
    return m

approx_results = {}
for name, builder in approximator_builders.items():
    print(f"\nTraining {name} on synapse data")
    model = builder()
    if name == 'RNN_Seq':
        history = model.fit(
            X_train, y_train,
            epochs=100, batch_size=32,
            validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr], verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=10000, batch_size=64,
            validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr], verbose=1
        )
    loss = model.evaluate(X_test, y_test, verbose=0)
    approx_results[name] = {'loss': loss, 'model': model, 'history': history.history}
    print(f"Test loss for {name}: {loss:.6f}")
print("\nAll approximators trained on synapse dataset. Summary:")
for name, r in approx_results.items():
    print(f"  {name}: {r['loss']:.6f}")

# Visualize predictions for each approximator
n_vis_samples = 10
for name, info in approx_results.items():
    model = info['model']
    print(f"\nPlotting predictions for {name}")
    for i in range(n_vis_samples):
        sample_idx = np.random.randint(0, X_test.shape[0])
        input_sample = X_test[sample_idx:sample_idx+1]
        true_scaled = y_test[sample_idx]
        # predict sequence
        pred_seq = model.predict(input_sample)
        # reshape to (SEQUENCE_LENGTH, 1)
        pred_scaled = pred_seq.reshape(SEQUENCE_LENGTH, 1)
        # inverse transform
        true_orig = scaler_y.inverse_transform(true_scaled.reshape(-1,1))
        pred_orig = scaler_y.inverse_transform(pred_scaled)
        time_axis = np.arange(SEQUENCE_LENGTH) * (b2.defaultclock.dt / b2.ms)
        # plot
        fig, ax1 = plt.subplots(figsize=(15,7))
        ax1.set_xlabel(f'Time (ms), dt={b2.defaultclock.dt/b2.ms:.2f} ms')
        ax1.set_ylabel('Input Spikes', color='tab:red')
        spikes = np.where(input_sample[0,:,0]>0.5)[0]*(b2.defaultclock.dt/b2.ms)
        if spikes.size>0:
            marker, stems, base = ax1.stem(spikes, np.ones_like(spikes), linefmt='-', markerfmt='o', basefmt=' ')
            plt.setp(marker, color='tab:red'); plt.setp(stems, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.set_ylim(-0.1,1.5)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Conductance', color='tab:blue')
        ax2.plot(time_axis, true_orig[:,0], color='tab:blue', label='True')
        ax2.plot(time_axis, pred_orig[:,0], color='tab:green', linestyle='--', label='Predicted')
        plt.title(f"{name} Prediction vs True (Sample {sample_idx})")
        fig.tight_layout(); plt.legend(); plt.show()


