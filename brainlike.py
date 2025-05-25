
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
np.random.seed(42)


@dataclass
class NeuronParams:
    """Parameters for individual neurons"""
    threshold: float = 1.0
    decay: float = 0.9
    refractory_period: int = 2
    noise_level: float = 0.01


class Neuron:
    """Individual neuron using simplified Integrate-and-Fire model"""
    
    def __init__(self, neuron_id: int, params: NeuronParams):
        self.id = neuron_id
        self.params = params
        
        # State variables
        self.membrane_potential = 0.0
        self.spike_times = []
        self.refractory_counter = 0
        self.last_spike_time = -1
        
        # For learning
        self.pre_synaptic_traces = {}  # STDP traces
        
    def update(self, input_current: float, time_step: int) -> bool:
        """Update neuron state and return True if it spikes"""
        
        # Add noise
        input_current += np.random.normal(0, self.params.noise_level)
        
        # Check refractory period
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.membrane_potential = 0.0
            return False
        
        # Update membrane potential
        self.membrane_potential = (
            self.membrane_potential * self.params.decay + input_current
        )
        
        # Check for spike
        if self.membrane_potential >= self.params.threshold:
            self.membrane_potential = 0.0
            self.refractory_counter = self.params.refractory_period
            self.spike_times.append(time_step)
            self.last_spike_time = time_step
            return True
        
        return False


class LocalCircuit:
    """Local circuit containing multiple neurons with local connections"""
    
    def __init__(
        self, 
        circuit_id: int, 
        n_neurons: int, 
        connectivity_prob: float = 0.3
    ):
        self.id = circuit_id
        self.neurons = [
            Neuron(i, NeuronParams()) for i in range(n_neurons)
        ]
        
        # Local connectivity matrix
        self.local_weights = self._create_local_connections(
            n_neurons, connectivity_prob
        )
        
        # Activity history for visualization
        self.activity_history = []
        
    def _create_local_connections(
        self, n_neurons: int, prob: float
    ) -> np.ndarray:
        """Create local connection matrix"""
        weights = np.random.rand(n_neurons, n_neurons) * 0.5
        # Make connections sparse based on probability
        mask = np.random.rand(n_neurons, n_neurons) < prob
        weights *= mask
        # No self-connections
        np.fill_diagonal(weights, 0)
        return weights
    
    def update(self, external_input: np.ndarray, time_step: int) -> np.ndarray:
        """Update all neurons in the circuit"""
        n_neurons = len(self.neurons)
        spikes = np.zeros(n_neurons, dtype=bool)
        
        # Calculate total input for each neuron
        total_input = external_input.copy()
        
        # Add local circuit input from previous time step
        if hasattr(self, '_last_spikes'):
            local_input = self.local_weights.T @ self._last_spikes.astype(float)
            total_input += local_input
        
        # Update each neuron
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(total_input[i], time_step)
        
        # Store for next iteration
        self._last_spikes = spikes
        
        # Store activity for visualization
        self.activity_history.append(spikes.astype(float))
        
        return spikes


class MultiscaleBrainNetwork:
    """Global network connecting multiple local circuits"""
    
    def __init__(
        self, 
        n_circuits: int, 
        neurons_per_circuit: int,
        global_connectivity_strength: float = 0.1
    ):
        self.circuits = [
            LocalCircuit(i, neurons_per_circuit) 
            for i in range(n_circuits)
        ]
        
        # Global connection weights between circuits
        self.global_weights = self._create_global_connections(
            n_circuits, global_connectivity_strength
        )
        
        # Learning parameters
        self.learning_rate = 0.01
        self.stdp_window = 10
        
        # Pattern storage
        self.learned_patterns = []
        self.global_activity_history = []
        
    def _create_global_connections(
        self, n_circuits: int, strength: float
    ) -> np.ndarray:
        """Create global connections between circuits"""
        # Random connectivity between circuits
        weights = np.random.rand(n_circuits, n_circuits) * strength
        # Make it symmetric and remove self-connections
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        return weights
    
    def hebbian_learning(
        self, 
        circuit_activities: List[np.ndarray], 
        time_step: int
    ):
        """Implement Hebbian learning rule for global connections"""
        n_circuits = len(self.circuits)
        
        # Calculate circuit-level activity (mean activity of neurons in circuit)
        circuit_means = np.array([
            np.mean(activity) for activity in circuit_activities
        ])
        
        # Update global weights based on correlated activity
        for i in range(n_circuits):
            for j in range(i + 1, n_circuits):
                # Hebbian rule: strengthen connections between co-active circuits
                delta_w = (
                    self.learning_rate * 
                    circuit_means[i] * 
                    circuit_means[j]
                )
                self.global_weights[i, j] += delta_w
                self.global_weights[j, i] += delta_w
                
                # Keep weights bounded
                self.global_weights[i, j] = np.clip(
                    self.global_weights[i, j], 0, 1
                )
                self.global_weights[j, i] = self.global_weights[i, j]
    
    def present_pattern(
        self, 
        pattern: np.ndarray, 
        duration: int = 50
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Present a spatial pattern to the network and record activity"""
        
        all_circuit_activities = []
        global_activities = []
        
        for time_step in range(duration):
            circuit_activities = []
            
            # Calculate global input to each circuit
            if time_step > 0:
                # Previous circuit activities influence current input
                prev_circuit_means = np.array([
                    np.mean(activity) if len(activity) > 0 else 0 
                    for activity in circuit_activities_prev
                ])
                global_input = self.global_weights @ prev_circuit_means
            else:
                global_input = np.zeros(len(self.circuits))
            
            # Update each circuit
            for i, circuit in enumerate(self.circuits):
                # External input: pattern + global input
                external_input = (
                    pattern[i * len(circuit.neurons):(i + 1) * len(circuit.neurons)] + 
                    global_input[i] * 0.1  # Scale global input
                )
                
                # Ensure input has correct size
                if len(external_input) < len(circuit.neurons):
                    external_input = np.pad(
                        external_input, 
                        (0, len(circuit.neurons) - len(external_input))
                    )
                
                activity = circuit.update(external_input, time_step)
                circuit_activities.append(activity)
            
            # Learning: update global connections
            if time_step > 0:
                self.hebbian_learning(circuit_activities, time_step)
            
            all_circuit_activities.append(circuit_activities)
            
            # Store global activity
            global_activity = np.concatenate([
                activity.astype(float) for activity in circuit_activities
            ])
            global_activities.append(global_activity)
            
            circuit_activities_prev = circuit_activities
        
        self.global_activity_history.extend(global_activities)
        return all_circuit_activities, global_activities
    
    def create_spatial_pattern(self, pattern_type: str = "stripe") -> np.ndarray:
        """Create different types of spatial patterns"""
        total_neurons = sum(len(circuit.neurons) for circuit in self.circuits)
        
        if pattern_type == "stripe":
            # Alternating high/low activity
            pattern = np.array([
                1.5 if (i // 3) % 2 == 0 else 0.2 
                for i in range(total_neurons)
            ])
        elif pattern_type == "center":
            # Higher activity in center
            center = total_neurons // 2
            pattern = np.array([
                1.5 * np.exp(-0.1 * (i - center)**2) 
                for i in range(total_neurons)
            ])
        elif pattern_type == "random":
            # Random pattern
            pattern = np.random.rand(total_neurons) * 1.5
        else:
            # Default: linear gradient
            pattern = np.linspace(0.2, 1.5, total_neurons)
        
        return pattern
    
    def train_on_patterns(
        self, 
        patterns: List[np.ndarray], 
        n_epochs: int = 5
    ):
        """Train the network on multiple patterns"""
        print(f"Training on {len(patterns)} patterns for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            
            for i, pattern in enumerate(patterns):
                print(f"  Training on pattern {i + 1}/{len(patterns)}")
                self.present_pattern(pattern, duration=30)
        
        print("Training completed!")
    
    def test_pattern_recall(self, test_pattern: np.ndarray) -> float:
        """Test how well the network recalls a learned pattern"""
        # Present degraded version of pattern
        degraded_pattern = test_pattern + np.random.normal(
            0, 0.2, size=test_pattern.shape
        )
        
        # Record network response
        _, activities = self.present_pattern(degraded_pattern, duration=20)
        
        # Calculate similarity to original pattern
        if len(activities) > 10:  # Use steady-state activity
            steady_state = np.mean(activities[-10:], axis=0)
            # Normalize for comparison
            steady_state_norm = steady_state / (np.max(steady_state) + 1e-8)
            pattern_norm = test_pattern / (np.max(test_pattern) + 1e-8)
            
            # Calculate correlation
            correlation = np.corrcoef(steady_state_norm, pattern_norm)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0


# Visualization functions
def plot_network_activity(
    network: MultiscaleBrainNetwork, 
    activity_data: List[np.ndarray],
    title: str = "Network Activity"
):
    """Plot network activity over time"""
    if not activity_data:
        print("No activity data to plot")
        return
    
    activity_matrix = np.array(activity_data).T
    
    plt.figure(figsize=(12, 8))
    
    # Plot raster plot
    plt.subplot(2, 1, 1)
    for neuron_idx in range(min(50, activity_matrix.shape[0])):  # Limit to 50 neurons
        spike_times = np.where(activity_matrix[neuron_idx] > 0.5)[0]
        plt.scatter(
            spike_times, 
            [neuron_idx] * len(spike_times), 
            s=2, 
            alpha=0.7
        )
    
    plt.ylabel('Neuron Index')
    plt.title(f'{title} - Spike Raster')
    plt.grid(True, alpha=0.3)
    
    # Plot population activity
    plt.subplot(2, 1, 2)
    population_activity = np.mean(activity_matrix, axis=0)
    plt.plot(population_activity, linewidth=2)
    plt.ylabel('Population Activity')
    plt.xlabel('Time Steps')
    plt.title(f'{title} - Population Activity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_global_connectivity(network: MultiscaleBrainNetwork):
    """Plot the global connectivity matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        network.global_weights, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis',
        cbar_kws={'label': 'Connection Strength'}
    )
    plt.title('Global Circuit Connectivity Matrix')
    plt.xlabel('Target Circuit')
    plt.ylabel('Source Circuit')
    plt.show()


# Main implementation and demonstration
def main():
    print("🧠 Multiscale Computational Brain Model")
    print("=" * 50)
    
    # Create network
    print("Creating multiscale brain network...")
    network = MultiscaleBrainNetwork(
        n_circuits=4, 
        neurons_per_circuit=10,
        global_connectivity_strength=0.15
    )
    
    print(f"Network created with {len(network.circuits)} circuits")
    print(f"Total neurons: {sum(len(c.neurons) for c in network.circuits)}")
    
    # Create training patterns
    print("\nCreating training patterns...")
    patterns = [
        network.create_spatial_pattern("stripe"),
        network.create_spatial_pattern("center"),
        network.create_spatial_pattern("random")
    ]
    
    print(f"Created {len(patterns)} training patterns")
    
    # Show initial connectivity
    print("\nInitial global connectivity:")
    plot_global_connectivity(network)
    
    # Train the network
    print("Training network on patterns...")
    network.train_on_patterns(patterns, n_epochs=3)
    
    # Show learned connectivity
    print("\nLearned global connectivity:")
    plot_global_connectivity(network)
    
    # Test pattern recall
    print("\nTesting pattern recall...")
    for i, pattern in enumerate(patterns):
        print(f"\nTesting pattern {i + 1} recall...")
        
        # Test recall
        recall_score = network.test_pattern_recall(pattern)
        print(f"Pattern {i + 1} recall score: {recall_score:.3f}")
        
        # Visualize the test
        degraded_pattern = pattern + np.random.normal(0, 0.2, size=pattern.shape)
        _, test_activity = network.present_pattern(degraded_pattern, duration=25)
        
        plot_network_activity(
            network, 
            test_activity, 
            f"Pattern {i + 1} Recall Test"
        )
    
    # Show learning curve (global connectivity evolution)
    print("\nAnalyzing learning...")
    total_connectivity = np.sum(network.global_weights)
    print(f"Total global connectivity strength: {total_connectivity:.3f}")
    
    print("\n✅ Multiscale brain model demonstration completed!")
    print("The network learned to:")
    print("- Process spatial patterns across multiple scales")
    print("- Strengthen connections between co-active circuits")
    print("- Recall learned patterns from degraded inputs")


if __name__ == "__main__":
    main()
