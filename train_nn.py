'''
/******************/
/*  train_nn.py   */
/*  Version 1.0   */
/*   2025/04/27   */
/******************/
'''
import argparse
import numpy as np
import os
import sys
import shutil

from mod_config import nn_config as cfg, game_cfg
from game_logic import GameLogic

cwd = os.getcwd()
build_dir = os.path.join(cwd, "./externals/ma-libs/build")

if "DEBUG" in os.environ:
    build_path = os.path.join(build_dir, "Debug")
else:
    build_path = os.path.join(build_dir, "Release")

# Add the appropriate build path to sys.path
sys.path.append(os.path.realpath(build_path))

# Try importing the cpp_nn_py module
try:
    import cpp_nn_py as cpp_nn_py
except ModuleNotFoundError as e:
    print(f"Error: {e}")


class NeuralNetwork():
    def __init__(self):
        self._net = None
        self._nGen = 0
        self._nnsize = None
        self._game_env = None  # Can be initialized later if needed per member
        pass

    def init(self):
        if cfg.save_nn:
            # create directory if it doesn't exist
            os.makedirs(cfg.save_path_nn, exist_ok=True)
        # GameLogic expects 7 inputs and 4 outputs
        self._nnsize = [7] + cfg.hlayers + [4]

        self._net = cpp_nn_py.ANN_MLP_GA_double(
            self._nnsize, cfg.seed, cfg.population_size,
            cfg.top_individuals, cfg.activation_id, cfg.elitism)

        self._net.SetName(cfg.name)
        self._net.SetMixed(cfg.mixed_population)
        self._net.CreatePopulation(True)
        self._nGen = 0  # Start from generation 0 if initializing

    def load(self, step='last'):
        self._net = cpp_nn_py.ANN_MLP_GA_double()
        self._net.SetName(cfg.name)
        if step == 'last':
            filename = f"{cfg.name}_last.hd5"
        else:
            filename = f"{cfg.name}_{step}.hd5"

        full_path = os.path.join(cfg.save_path_nn, filename)
        self._net.Deserialize(full_path)
        # Get current epoch/generation from loaded net
        self._nGen = self._net.GetEpochs()
        # GameLogic expects 7 inputs and 4 outputs
        self._nnsize = [7] + cfg.hlayers + [4]
        if cfg.verbose:
            print(f"Loading network from: {full_path} "
                  f"(Generation {self._nGen})")

    def save(self):
        """Saves the network state and creates a _last copy."""
        # Construct the base filename
        base_filename_part = f"{cfg.name}"
        if not cfg.overwrite:
            base_filename_part += f"_{self._nGen}"
        filename = f"{base_filename_part}.hd5"
        full_path = os.path.join(cfg.save_path_nn, filename)

        # Construct the "_last" filename
        last_filename = f"{cfg.name}_last.hd5"
        last_full_path = os.path.join(cfg.save_path_nn, last_filename)

        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError as e:
                print(f"Error removing existing file {full_path}: {e}")

        # Serialize the network
        try:
            self._net.Serialize(full_path)
            if cfg.verbose:
                print(f"Network serialized to: {full_path}")
            # Create the "_last" copy
            try:
                shutil.copy2(full_path, last_full_path)
                if cfg.verbose:
                    print(f"Copied network state to: {last_full_path}")
            except OSError as e:
                print(f"Error copying {full_path} to {last_full_path}: {e}")

        except Exception as e:
            print(f"Error during network serialization to {full_path}: {e}")

    def _calculate_fitness(self, game_sim: GameLogic, steps_taken: int) \
            -> float:
        """Calculates fitness score for a completed game simulation.
           Lower score is better.
        """
        fitness = 0.0

        # Base penalty for steps taken (encourages efficiency)
        fitness += steps_taken * 0.1

        # Distance penalty (applied more heavily at the end)
        dist_x = abs(game_sim.x - game_sim.landing_pad_center_x)
        dist_y = abs(game_sim.y - game_sim.landing_pad_y)
        # Scale distance penalty - higher if further away
        fitness += (dist_x + dist_y) * 0.5

        if game_sim.landed_successfully:
            fitness -= 1000.0  # Big reward (negative penalty)
            # Bonus for remaining fuel
            fitness -= game_sim.fuel * 2.0
        elif game_sim.crashed:
            fitness += 500.0  # Penalty for crashing
            # Increase penalty based on final velocity magnitude
            final_v_mag = np.sqrt(game_sim.vx**2 + game_sim.vy**2)
            fitness += final_v_mag * 10.0
        elif game_sim.fuel <= 0 and not game_sim.landed:
            fitness += 250.0  # Penalty for running out of fuel mid-air

        # Add small penalty based on final velocity if not landed successfully
        if not game_sim.landed_successfully:
            final_v_mag = np.sqrt(game_sim.vx**2 + game_sim.vy**2)
            fitness += final_v_mag * 1.0

        return fitness

    def train(self):
        if self._net is None:
            print("Error: Network not initialized or loaded.")
            return

        print(f"Starting training for {cfg.epochs} generations...")
        print(f"Population size: {cfg.population_size}, "
              f"Top performers: {cfg.top_individuals}")
        print(f"Network structure: {self._nnsize}")

        num_outputs = self._nnsize[-1]
        population_size = self._net.GetPopSize()  # Use getter

        for gen in range(self._nGen, self._nGen + cfg.epochs):
            fitness_scores = np.zeros(population_size, dtype=np.float64)
            all_steps = []  # Track steps per member for info

            print(f"\n--- Generation {gen + 1} ---")

            for member_id in range(population_size):
                game_sim = GameLogic(no_print=True)
                state = game_sim.get_state()
                done = False
                current_fitness_penalty = 0.0  # Accumulate penalties
                steps = 0

                while not done and steps < game_cfg.max_steps:
                    inputs = np.array(state, dtype=np.float64)
                    outputs = np.zeros(num_outputs, dtype=np.float64)

                    # Get action from NN
                    self._net.feedforward(inputs, outputs, member_id, False)
                    action = np.argmax(outputs)

                    # Update game state
                    next_state, reward, done = game_sim.update(action)

                    # --- Optional: Add small penalties during the run ---
                    # Penalty for distance from pad center
                    dist_x = abs(game_sim.x - game_sim.landing_pad_center_x)
                    dist_y = abs(game_sim.y - game_sim.landing_pad_y)
                    current_fitness_penalty += (dist_x + dist_y) * 0.001
                    # Penalty for using fuel (if action > 0)
                    # current_fitness_penalty += (action > 0) * 0.01
                    # ----------------------------------------------------

                    state = next_state
                    steps += 1

                    if done:
                        break

                # Calculate final fitness after episode ends
                final_fitness = self._calculate_fitness(game_sim, steps)
                fitness_scores[member_id] = final_fitness + \
                    current_fitness_penalty
                all_steps.append(steps)

            # Sort fitness scores (ascending, lower is better) and get indices
            sorted_indices = np.argsort(fitness_scores)

            # Update NN weights and biases based on sorted fitness
            self._net.UpdateWeightsAndBiases(sorted_indices)

            # Create the next population
            self._net.CreatePopulation(cfg.elitism)

            # Update internal epoch counter in the C++ object
            self._net.UpdateEpochs(1)  # Increment epoch by 1
            self._nGen = self._net.GetEpochs()  # Sync python counter

            # Print generation summary
            best_fitness = fitness_scores[sorted_indices[0]]
            avg_fitness = np.mean(fitness_scores)
            avg_steps = np.mean(all_steps)
            print(f"Generation {self._nGen} complete.")
            print(
                f"  Best Fitness: {best_fitness:.4f} (Member "
                f"{sorted_indices[0]})")
            print(f"  Avg Fitness:  {avg_fitness:.4f}")
            print(f"  Avg Steps:    {avg_steps:.1f}")

            # Save network periodically
            if cfg.save_nn and (self._nGen % cfg.save_interval == 0 or
                                gen == self._nGen + cfg.epochs - 1):
                self.save()

        print("\nTraining finished.")

    def get_action(self, current_state: np.ndarray) -> int:
        """Gets the action from the NN based on the current state."""
        if self._net is None:
            print("Error: Network not loaded or initialized.")
            # Return a default action (e.g., Noop) or raise an error
            return 0

        if self._nnsize is None:
            print("Error: Network size not determined (load network first).")
            return 0

        num_outputs = self._nnsize[-1]
        inputs = np.array(current_state, dtype=np.float64)
        outputs = np.zeros(num_outputs, dtype=np.float64)

        # Use member_id 0 - assuming the loaded network represents the best
        # or the result of the population evolution.
        # The GA manages this internally.
        # singleReturn=False as per the pybind definition
        try:
            self._net.feedforward(inputs, outputs, 0, False)
            action = np.argmax(outputs)
            return action
        except Exception as e:
            print(f"Error during feedforward: {e}")
            return 0  # Return default action on error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue', dest="cont", action='store_true',
        default=False, help="Continue training from checkpoint")
    parser.add_argument(
        '--step', type=int, default=0, help="Checkpoint step to load")
    args = parser.parse_args()

    NN = NeuralNetwork()

    if args.cont:
        NN.load(args.step)
    else:
        NN.init()
    NN.train()  # Start training
