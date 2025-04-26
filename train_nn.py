'''
/******************/
/*  train_nn.py   */
/*  Version 1.0   */
/*   2025/04/27   */
/******************/
'''
import numpy as np
from game_logic import GameLogic

# --- Placeholder for NN Interaction ---
# These functions will interact with your C++ NN library eventually


def initialize_nn():
    """Placeholder to initialize the Neural Network."""
    print("NN Placeholder: Initializing Network...")
    # TODO: Load or initialize your C++ model here
    pass


def get_nn_action(state: np.ndarray) -> int:
    """Placeholder function to get action from the NN."""
    # TODO: Replace with actual NN inference call using the state vector
    # The state vector format is defined in GameLogic.get_state()
    # Expected output: An integer action (0: Noop, 1: Up, 2: Left, 3: Right)
    # print(f"NN Placeholder: Received state shape: {state.shape}") # Debug
    action = np.random.randint(0, 4)  # Example: random action
    # print(f"NN Placeholder: Returning action: {action}") # Debug
    return action


def train_nn_batch(batch):
    """Placeholder function to train the NN on a batch of experiences."""
    # TODO: Implement the training step using data from the replay buffer
    print(f"NN Placeholder: Training on batch of size {len(batch)}...")
    pass


def save_nn_model(path="lander_model.pth"):
    """Placeholder function to save the trained NN model."""
    # TODO: Implement saving the model state
    print(f"NN Placeholder: Saving model to {path}...")
    pass
# --- End Placeholder ---


def run_training_loop():
    """Runs the NN training process (no GUI)."""
    print("\n--- Starting NN Training Mode (No GUI) ---")
    logic = GameLogic()
    initialize_nn()  # Initialize NN model/library

    # --- Training Parameters (Example) ---
    num_episodes = 50  # Number of episodes to train
    max_steps_per_episode = 1000  # Max steps before ending episode
    replay_buffer_capacity = 10000  # Max experiences to store
    batch_size = 64  # Number of experiences to sample for training
    training_start_buffer_size = 1000  # Start training only after buffer
    # --- End Training Parameters ---

    replay_buffer = []  # Simple list as replay buffer for now

    for episode in range(num_episodes):
        logic.reset()
        state = logic.get_state()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # 1. Get action from NN
            # Add exploration (e.g., epsilon-greedy)
            epsilon = max(0.01, 0.99 ** episode)  # Example epsilon decay
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)  # Explore: random action
            else:
                action = get_nn_action(state)  # Exploit: use NN prediction

            # 2. Update game logic
            next_state, reward, done = logic.update(action)
            total_reward += reward

            # 3. Store experience in replay buffer
            experience = (state, action, reward, next_state, done)
            replay_buffer.append(experience)
            # Keep buffer size limited
            if len(replay_buffer) > replay_buffer_capacity:
                replay_buffer.pop(0)  # Remove oldest experience

            # 4. Perform training step (if buffer is large enough)
            if len(replay_buffer) >= training_start_buffer_size and \
                    step % 4 == 0:  # Train every few steps
                # Sample a mini-batch from the replay buffer
                batch_indices = np.random.choice(len(replay_buffer),
                                                 batch_size, replace=False)
                mini_batch = [replay_buffer[i] for i in batch_indices]
                train_nn_batch(mini_batch)  # Train on the sampled batch

            state = next_state
            step += 1

        print(f"Episode {episode+1}/{num_episodes} finished after {step}"
              f"steps. Total Reward: {total_reward:.2f}, Epsilon: "
              f"{epsilon:.3f}")
        render_info = logic.get_render_info()
        print(f"  Result: Landed={render_info['landed']}, "
              f"Crashed={render_info['crashed']}, "
              f"Success={render_info['landed_successfully']}")

    print("--- Training Finished ---")
    save_nn_model()  # Save the trained model


if __name__ == '__main__':
    # Allow running training directly
    run_training_loop()
