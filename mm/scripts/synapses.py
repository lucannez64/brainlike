
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Brian2 Simulation Parameters
b2.defaultclock.dt = 0.1 * b2.ms  # Simulation time step
DURATION = 200 * b2.ms          # Duration of each simulation run
N_SAMPLES = 1000                # Number of data samples to generate
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
        net.run(DURATION, report='text') # report=None can speed it up slightly for many runs

        all_input_spikes[i, :] = input_spike_train.astype(float)
        recorded_g = state_mon.g_total[0]
        if len(recorded_g) < seq_len:
            padding = np.zeros(seq_len - len(recorded_g))
            all_output_conductances[i, :] = np.concatenate((recorded_g, padding))
        else:
            all_output_conductances[i, :] = recorded_g[:seq_len]

    return all_input_spikes, all_output_conductances

# Generate data
print("Generating training data with Brian2...")
# Use a smaller dt for Brian2 simulation for accuracy, then potentially downsample for CNN
# Or ensure CNN sequence length matches Brian2 steps
actual_dt_for_generation = b2.defaultclock.dt # This has units, e.g., 0.1*ms
print(f"Using actual_dt_for_generation: {actual_dt_for_generation}")
input_spikes, output_conductances = generate_synaptic_data(
    N_SAMPLES, SEQUENCE_LENGTH, actual_dt_for_generation
)
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
ax1.stem(spike_times_plot, np.ones_like(spike_times_plot), linefmt=color+'-', markerfmt=color+'o', basefmt=" ")
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


def build_cnn_model(input_shape, num_conv_layers=2, filters=32, kernel_size=5):
    """Builds a 1D CNN model for sequence-to-sequence prediction."""
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers
    for i in range(num_conv_layers):
        model.add(layers.Conv1D(
            filters=filters * (i + 1), # Increasing filters
            kernel_size=kernel_size,
            activation='relu',
            padding='same' # 'causal' for strict step-by-step prediction
        ))
        # Optional: BatchNormalization, Dropout, MaxPooling1D
        model.add(layers.BatchNormalization())
        if i < num_conv_layers -1 : # No pooling after last conv before output usually
             model.add(layers.MaxPooling1D(pool_size=2, padding='same'))


    # Output layer: We want an output of the same sequence length
    # A Conv1D layer with 1 filter can achieve this.
    # Or, if pooling reduced dimensionality too much, you might need UpSampling1D
    # or ensure pooling is compatible with desired output length.
    # If MaxPooling was used, the sequence length is reduced.
    # We need to ensure the output layer can produce the original sequence length.
    # For simplicity, if pooling is used, we might need to adjust.
    # Let's try a final Conv1D to map to the output channels.
    # If MaxPooling was used, the spatial dimension is halved each time.
    # We need to be careful here. Let's make pooling optional or adjust.

    # To ensure output is same length as input when using 'same' padding and no striding
    # in conv, and if pooling is used, we need to upsample or use a different strategy.
    # For now, let's assume we want to predict the full sequence.
    # If MaxPooling was used, we need to upsample back.
    # Example: if 1 maxpool layer with pool_size=2, we need UpSampling1D(2)
    
    # Let's try a model structure that maintains sequence length or reconstructs it
    # For simplicity, let's try a few Conv1D layers and then a final Conv1D for output.
    # If using MaxPooling, you'd typically have Dense layers or an RNN at the end,
    # or use UpSampling. For seq-to-seq with CNNs, often an encoder-decoder
    # structure or fully convolutional networks are used.

    # Simpler approach for now: Conv layers with 'same' padding, no pooling,
    # then a final Conv1D for output.
    
    # Re-thinking the simple CNN structure for seq-to-seq:
    model_s2s = keras.Sequential(name=f"CNN_{num_conv_layers}_layers")
    model_s2s.add(layers.Input(shape=input_shape))
    for _ in range(num_conv_layers):
        model_s2s.add(layers.Conv1D(filters, kernel_size, activation='relu', padding='same'))
        model_s2s.add(layers.BatchNormalization()) # Good practice
    # Final Conv1D to map to the single output feature (conductance) at each time step
    model_s2s.add(layers.Conv1D(1, kernel_size=1, activation='linear', padding='same')) # Linear for regression

    model_s2s.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    model_s2s.summary()
    return model_s2s

# --- Experiment with number of layers ---
# Minimal could be 1 Conv1D layer + Output Conv1D layer
# Let's try with 2 hidden Conv1D layers first.
# The input_shape for Conv1D is (sequence_length, num_features_per_step)
cnn_input_shape = (SEQUENCE_LENGTH, 1)

# Try different numbers of layers
results = {}

for num_layers_exp in [1, 2, 3]: # Number of hidden Conv1D layers
    print(f"\n--- Training CNN with {num_layers_exp} hidden Conv1D layer(s) ---")
    model = build_cnn_model(cnn_input_shape, num_conv_layers=num_layers_exp, filters=32, kernel_size=7)

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=50, # Adjust as needed
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    loss = model.evaluate(X_test, y_test, verbose=0)
    results[num_layers_exp] = {'loss': loss, 'model': model, 'history': history.history}
    print(f"Test MSE for {num_layers_exp} hidden layer(s): {loss:.6f}")

# Find best model based on test loss
best_num_layers = min(results, key=lambda k: results[k]['loss'])
best_model = results[best_num_layers]['model']
best_loss = results[best_num_layers]['loss']
print(f"\nBest model has {best_num_layers} hidden layer(s) with Test MSE: {best_loss:.6f}")


# --- Plot training history for the best model ---
plt.figure(figsize=(10, 4))
plt.plot(results[best_num_layers]['history']['loss'], label='Train Loss')
plt.plot(results[best_num_layers]['history']['val_loss'], label='Validation Loss')
plt.title(f'Model Loss (Best: {best_num_layers} hidden layers)')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# --- Visualize Predictions of the best model ---
n_vis_samples = 3
for i in range(n_vis_samples):
    sample_idx_test = np.random.randint(0, X_test.shape[0])
    input_sample = X_test[sample_idx_test:sample_idx_test+1]
    true_output_scaled = y_test[sample_idx_test]
    
    predicted_output_scaled = best_model.predict(input_sample)[0]

    # Inverse transform to original scale for interpretation
    true_output_original = scaler_y.inverse_transform(true_output_scaled)
    predicted_output_original = scaler_y.inverse_transform(predicted_output_scaled)
    
    time_axis = np.arange(SEQUENCE_LENGTH) * (b2.defaultclock.dt / b2.ms)

    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # Plot input spikes
    color = 'tab:red'
    ax1.set_xlabel(f'Time (ms), dt={b2.defaultclock.dt / b2.ms:.2f} ms')
    ax1.set_ylabel('Input Spikes', color=color)
    spike_times_plot = np.where(input_sample[0, :, 0] > 0.5)[0] * (b2.defaultclock.dt / b2.ms)
    if len(spike_times_plot) > 0:
        ax1.stem(spike_times_plot, np.ones_like(spike_times_plot), linefmt=color+'-', markerfmt=color+'o', basefmt=" ")
    else: # Plot a flat line if no spikes, so ylim is consistent
        ax1.plot([time_axis[0], time_axis[-1]], [0,0], color=color, alpha=0) # Invisible line for ylim
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.1, 1.5) # Adjusted for stem plot base

    # Plot conductances
    ax2 = ax1.twinx()
    color_true = 'tab:blue'
    color_pred = 'tab:green'
    
    ax2.set_ylabel('Synaptic Conductance (Original Scale)', color=color_true)
    ax2.plot(time_axis, true_output_original[:, 0], color=color_true, linestyle='-', label='Brian2 (True)')
    ax2.plot(time_axis, predicted_output_original[:, 0], color=color_pred, linestyle='--', label=f'CNN Prediction ({best_num_layers} layers)')
    ax2.tick_params(axis='y', labelcolor=color_true)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.title(f"Comparison on Test Sample {sample_idx_test} (Original Scale)")
    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()
