
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Optional
from itertools import product
import random
import time

np.random.seed(int(time.time())) # More randomness for RL

def sigmoid(x):
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x_activated):
    return x_activated * (1 - x_activated)

class MicroCircuit: # Unchanged from previous
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

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int) # 0: empty, 1: P1 (AI), -1: P2 (Opponent)
        self.current_player = 1 # AI starts
        self.winner = None

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.winner = None
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_valid_actions(self) -> List[int]:
        return [i for i, val in enumerate(self.board) if val == 0]

    def check_winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
            [0, 4, 8], [2, 4, 6]             # diagonals
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != 0:
                self.winner = self.board[line[0]]
                return self.winner
        if not self.get_valid_actions(): # No empty squares
            self.winner = 0 # Draw
            return 0
        return None # Game ongoing

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.board[action] != 0:
            # Invalid move - should be handled by agent's valid action selection
            # Penalize heavily and end game for simplicity here, or raise error
            return self.get_state(), -10.0, True, {"error": "Invalid move"} 

        self.board[action] = self.current_player
        
        winner = self.check_winner()
        done = winner is not None
        reward = 0.0

        if done:
            if winner == 1: reward = 1.0  # AI wins
            elif winner == -1: reward = -1.0 # Opponent wins
            # Draw reward is 0 by default
        # else:
            # reward = -0.01 # Small penalty for each move to encourage faster wins (optional)

        self.current_player *= -1 # Switch player
        return self.get_state(), reward, done, {}

    def print_board(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for i in range(3):
            print(" ".join(symbols[self.board[j]] for j in range(i*3, (i+1)*3)))
        print("-" * 5)

class QLearningAgent:
    def __init__(
        self, 
        n_inputs: int = 9, # Board size
        n_outputs: int = 9, # Q-value for each square
        n_hidden_circuits: int = 10, # Increased capacity
        n_internal_units_per_circuit: int = 4,
        learning_rate: float = 0.1, # RL often needs smaller LR
        gamma: float = 0.95, # Discount factor
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_circuits = n_hidden_circuits
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.hidden_circuits = [
            MicroCircuit(n_internal_units_per_circuit, input_scale=1.0) 
            for _ in range(n_hidden_circuits)
        ]
        
        limit_w1 = np.sqrt(6 / (n_inputs + n_hidden_circuits))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, (n_inputs, n_hidden_circuits))
        self.b1 = np.zeros(n_hidden_circuits)
        
        limit_w2 = np.sqrt(6 / (n_hidden_circuits + n_outputs)) # n_outputs instead of 1
        self.W2 = np.random.uniform(-limit_w2, limit_w2, (n_hidden_circuits, n_outputs))
        self.b2 = np.zeros(n_outputs) # Vector of biases for output layer

    def forward_pass(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # state is the board vector
        x = state.astype(np.float32) 
        hidden_circuit_inputs_linear = np.dot(x, self.W1) + self.b1
        
        hidden_circuit_outputs = np.zeros(self.n_hidden_circuits)
        circuit_internal_activations_cache = [] 

        for i, circuit in enumerate(self.hidden_circuits):
            scalar_input_to_circuit = hidden_circuit_inputs_linear[i]
            output, _, internal_acts = circuit.activate(scalar_input_to_circuit)
            hidden_circuit_outputs[i] = output
            circuit_internal_activations_cache.append(internal_acts)
            
        final_output_linear = np.dot(hidden_circuit_outputs, self.W2) + self.b2
        q_values = sigmoid(final_output_linear) # Q-values for all 9 actions
        
        cache = {
            'x': x,
            'hidden_circuit_inputs_linear': hidden_circuit_inputs_linear,
            'hidden_circuit_outputs': hidden_circuit_outputs,
            'circuit_internal_activations_cache': circuit_internal_activations_cache,
            'final_output_linear': final_output_linear,
            'q_values': q_values # Renamed from final_prediction
        }
        return q_values, cache

    def choose_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions) # Explore
        else:
            q_values, _ = self.forward_pass(state)
            # Select best action among valid ones
            valid_q_values = {action: q_values[action] for action in valid_actions}
            if not valid_q_values: # Should not happen if valid_actions is not empty
                 return random.choice(valid_actions)
            return max(valid_q_values, key=valid_q_values.get) # Exploit

    def backward_pass(self, target_q_for_action: float, chosen_action: int, cache: Dict):
        x = cache['x']
        z_h = cache['hidden_circuit_inputs_linear']
        a_h = cache['hidden_circuit_outputs']
        internal_acts_cache = cache['circuit_internal_activations_cache']
        q_values_activated = cache['q_values'] # These are sigmoid(z_o)
        
        # The error is only for the chosen action's output neuron
        # dL/dz_o for the chosen action. For others, it's 0.
        # L = 0.5 * (target_q - q_predicted)^2
        # dL/dq_predicted = q_predicted - target_q
        # dq_predicted/dz_o = sigmoid_derivative(q_predicted)
        # So, dL/dz_o = (q_predicted - target_q) * sigmoid_derivative(q_predicted)
        
        error_output_layer_dz = np.zeros_like(q_values_activated)
        q_predicted_for_action = q_values_activated[chosen_action]
        
        # Target Q needs to be in the same scale as sigmoid output (0,1)
        # If rewards are -1, 0, 1, we need to scale them or use linear output for Q.
        # For simplicity, let's assume target_q_for_action is already scaled (e.g. 0 for loss, 0.5 for draw, 1 for win)
        # Or, more directly, the error is (q_predicted - target_q)
        # And the gradient w.r.t. z_o is (q_predicted - target_q) * sigmoid_derivative(z_o)
        
        # Let's use the simpler dL/dz_o = (a_o - y) for the specific action
        # where a_o is q_values_activated[chosen_action] and y is target_q_for_action
        # This is the gradient of the output of the sigmoid w.r.t its input (z_o)
        # times the gradient of the loss w.r.t the output of the sigmoid.
        # dL/dz = dL/da * da/dz
        # dL/da = (a-y)
        # da/dz = sigmoid_derivative(a)
        # So dL/dz = (a-y) * sigmoid_derivative(a)

        # Let's adjust the target to be in [0,1] for sigmoid output
        # Win: 1.0, Loss: 0.0, Draw: 0.5
        scaled_target_q = (target_q_for_action + 1) / 2 # Maps [-1, 1] to [0, 1]

        # Error for the specific output neuron corresponding to chosen_action
        # This is dL/dz_o[chosen_action]
        error_for_chosen_action_output_unit = (q_predicted_for_action - scaled_target_q) * sigmoid_derivative(q_predicted_for_action)
        error_output_layer_dz[chosen_action] = error_for_chosen_action_output_unit
        
        dW2 = np.outer(a_h, error_output_layer_dz) # a_h is (n_hidden,), error is (n_outputs,)
        db2 = error_output_layer_dz
        
        error_propagated_to_hidden_outputs = np.dot(error_output_layer_dz, self.W2.T) # dL/da_h
        
        dL_dz_h = np.zeros_like(z_h)
        for i, circuit in enumerate(self.hidden_circuits):
            circuit_derivative = circuit.derivative_output_wrt_input(internal_acts_cache[i])
            dL_dz_h[i] = error_propagated_to_hidden_outputs[i] * circuit_derivative
            
        dW1 = np.outer(x, dL_dz_h)
        db1 = dL_dz_h
        
        clip_val = 0.5 # Smaller clip for RL
        dW1 = np.clip(dW1, -clip_val, clip_val)
        dW2 = np.clip(dW2, -clip_val, clip_val)
        db1 = np.clip(db1, -clip_val, clip_val)
        db2 = np.clip(db2, -clip_val, clip_val)
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def learn(self, state, action, reward, next_state, done):
        q_values_current_state, cache = self.forward_pass(state)
        q_predicted_for_action = q_values_current_state[action]

        # Target Q-value calculation
        if done:
            target_q_for_action = reward # Actual reward received
        else:
            q_values_next_state, _ = self.forward_pass(next_state)
            # For Q-learning, we need Q(s',a') for *opponent's* perspective if it's their turn
            # Or, if we model Q for current player, then max_q_next is for current player's next turn
            # Let's assume Q values are always for the current player whose turn it would be in that state.
            # Since the board state `next_state` reflects the board *after* AI's move,
            # if the game is not done, it's opponent's turn.
            # For simplicity, let's use max Q for the AI if it were to play in next_state.
            # This is a simplification; a more robust approach might involve minimax or opponent modeling.
            
            # Get valid actions for the *next* state (from AI's perspective if it were to play)
            # This is tricky because next_state is after AI's move, so it's opponent's turn.
            # A common simplification: Q(s,a) = r + gamma * max_a' Q(s',a') where s' is after AI's move.
            # The Q-values from self.forward_pass(next_state) are for the AI.
            
            # We need valid actions for the AI if it were to play from next_state.
            # However, the game env.get_valid_actions() is for the *actual* current player.
            # Let's assume the Q-network predicts values for the AI (player 1).
            # When calculating target, if next_state is opponent's turn, the value of that state for AI is
            # often taken as -max_a' Q_opponent(s',a') or similar.
            # For a simpler Q-learning, we just use max_a' Q_AI(s',a') from AI's network.
            
            # Get valid actions for the *next_state* (assuming it's AI's turn to evaluate possibilities)
            # This requires a temporary env or careful thought.
            # Let's assume the Q-values from forward_pass(next_state) are for the AI.
            # We need to filter these by what would be valid moves in next_state.
            temp_env = TicTacToeEnv()
            temp_env.board = next_state.copy() 
            # Whose turn is it in next_state? If AI just moved, it's opponent.
            # But Q-values are for AI. So, we need max Q if AI *could* move.
            # This is a common point of confusion. Let's assume Q(s,a) is value for player whose turn it is.
            # Our network is for player 1.
            # If player 1 made a move, next_state is for player -1.
            # So, target = r + gamma * (-max_a' Q_player1(s', a')) (if player -1 plays optimally to minimize player 1's Q)
            # This leads to Minimax Q-learning.
            
            # Simpler Q-learning: target = r + gamma * max_a' Q_player1(s', a')
            # This assumes the next state's value is determined by player 1's best move from there,
            # ignoring the opponent's turn. This is often sufficient for simple games.
            
            valid_actions_in_next_state = [i for i, val in enumerate(next_state) if val == 0]
            if not valid_actions_in_next_state: # Terminal state reached by opponent
                 max_q_next_state = 0.0 # No further moves for AI
            else:
                max_q_next_state = np.max(q_values_next_state[valid_actions_in_next_state])
            
            target_q_for_action = reward + self.gamma * max_q_next_state
        
        # The target_q_for_action is the "y" for the MSE loss for the chosen action.
        # It's already in the scale of rewards [-1, 1].
        self.backward_pass(target_q_for_action, action, cache)

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

# --- Opponent ---
class RandomOpponent:
    def choose_action(self, valid_actions: List[int]) -> Optional[int]:
        if not valid_actions:
            return None
        return random.choice(valid_actions)

# --- Training ---
def train_tic_tac_toe(agent: QLearningAgent, env: TicTacToeEnv, opponent: RandomOpponent, num_episodes: int = 20000):
    print(f"Training for {num_episodes} episodes...")
    wins = 0
    losses = 0
    draws = 0
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        current_episode_reward = 0

        while not done:
            if env.current_player == 1: # AI's turn
                valid_actions = env.get_valid_actions()
                if not valid_actions: break # Should not happen if game logic is correct
                action = agent.choose_action(state, valid_actions)
                
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done) # AI learns from its move
                state = next_state
                current_episode_reward += reward
            else: # Opponent's turn
                valid_actions = env.get_valid_actions()
                if not valid_actions: break
                opponent_action = opponent.choose_action(valid_actions)
                if opponent_action is None: break # Should not happen
                
                # AI doesn't learn from opponent's move directly in this simple Q-learning setup
                # The consequence of opponent's move is seen in the 'next_state' for AI's subsequent turn.
                state, reward_from_opp_move, done, _ = env.step(opponent_action)
                # current_episode_reward += reward_from_opp_move # Reward is from AI's perspective
        
        episode_rewards.append(current_episode_reward)
        if env.winner == 1: wins += 1
        elif env.winner == -1: losses += 1
        else: draws += 1

        if (episode + 1) % (num_episodes // 20) == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"W: {wins}, L: {losses}, D: {draws} (last 1000 avg rew: {np.mean(episode_rewards[-1000:]):.3f}) "
                  f"Epsilon: {agent.epsilon:.3f}")
            wins, losses, draws = 0,0,0 # Reset counts for next interval

    return episode_rewards


def play_game_with_ai(agent: QLearningAgent, env: TicTacToeEnv, opponent: RandomOpponent, ai_starts: bool = True):
    state = env.reset()
    if not ai_starts:
        env.current_player = -1 # Opponent starts
    
    done = False
    agent.epsilon = 0 # Pure exploitation for play

    print("\n--- New Game ---")
    env.print_board()

    while not done:
        if env.current_player == 1: # AI's turn
            print("AI's turn (X):")
            valid_actions = env.get_valid_actions()
            if not valid_actions: break
            action = agent.choose_action(state, valid_actions)
            print(f"AI chooses: {action}")
        else: # Opponent's turn
            print("Opponent's turn (O):")
            valid_actions = env.get_valid_actions()
            if not valid_actions: break
            action = opponent.choose_action(valid_actions)
            if action is None: break
            print(f"Opponent chooses: {action}")

        state, _, done, _ = env.step(action)
        env.print_board()
        if done:
            if env.winner == 1: print("AI (X) wins!")
            elif env.winner == -1: print("Opponent (O) wins!")
            else: print("It's a draw!")
            break
    agent.epsilon = agent.epsilon_start # Reset for potential further training


def main():
    print("🧠 ComplexLearner playing Tic-Tac-Toe with Q-Learning")
    print("=" * 60)

    env = TicTacToeEnv()
    agent = QLearningAgent(
        n_hidden_circuits=16, # More capacity for TTT
        n_internal_units_per_circuit=5,
        learning_rate=0.05, # Smaller LR for stability
        gamma=0.9,
        epsilon_decay=0.9995, # Slower decay
        epsilon_end=0.01
    )
    opponent = RandomOpponent()

    # Training
    num_training_episodes = 50000 # Needs significant training
    rewards = train_tic_tac_toe(agent, env, opponent, num_episodes=num_training_episodes)

    # Plot rewards
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.5, label='Raw Episode Reward')
    # Moving average
    window_size = num_training_episodes // 100
    if window_size > 0 :
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, color='red', label=f'Moving Avg (size {window_size})')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("AI Agent Training Rewards for Tic-Tac-Toe")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Play some games
    for _ in range(3):
        play_game_with_ai(agent, env, opponent, ai_starts=True)
    for _ in range(2):
        play_game_with_ai(agent, env, opponent, ai_starts=False)

if __name__ == "__main__":
    main()
