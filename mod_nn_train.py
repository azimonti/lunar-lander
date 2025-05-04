'''
/*******************/
/* mod_nn_train.py */
/*  Version 1.0    */
/*   2025/04/27    */
/*******************/
'''
import numpy as np
import os
import sys
import shutil

# Import necessary config objects and functions
# Alias nn_config to avoid conflict with the main nn_config from mod_config
from mod_config import cfg, nn_config, game_cfg, \
    reset_pad_positions, generate_random_pad_positions, set_pad_positions
from mod_game_logic import GameLogic

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
    import cpp_nn_py2 as cpp_nn_py
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
        if nn_config.save_nn:
            # create directory if it doesn't exist
            os.makedirs(nn_config.save_path_nn, exist_ok=True)
        # GameLogic expects 5 inputs and 4 outputs
        self._nnsize = [5] + nn_config.hlayers + [4]

        self._net = cpp_nn_py.ANN_MLP_GA_double(
            self._nnsize, nn_config.seed, nn_config.population_size,
            nn_config.top_individuals, nn_config.activation_id,
            nn_config.elitism)

        self._net.SetName(nn_config.name)
        self._net.SetMixed(nn_config.mixed_population)
        self._net.CreatePopulation(True)
        self._nGen = 0  # Start from generation 0 if initializing

    def load(self, step='last'):
        self._net = cpp_nn_py.ANN_MLP_GA_double()
        self._net.SetName(nn_config.name)
        if step == 'last':
            filename = f"{nn_config.name}_last.txt"
        else:
            filename = f"{nn_config.name}_{step}.txt"

        full_path = os.path.join(nn_config.save_path_nn, filename)
        self._net.Deserialize(full_path)
        # Get current epoch/generation from loaded net
        self._nGen = self._net.GetEpochs()
        # GameLogic expects 8 inputs and 4 outputs
        self._nnsize = [5] + nn_config.hlayers + [4]
        if nn_config.verbose:
            print(f"Loading network from: {full_path} "
                  f"(Generation {self._nGen})")
            print(f"  Expected network structure: {self._nnsize}")

    def save(self):
        """Saves the network state and creates a _last copy."""
        # Construct the base filename
        base_filename_part = f"{nn_config.name}"
        if not nn_config.overwrite:
            base_filename_part += f"_{self._nGen}"
        filename = f"{base_filename_part}.txt"
        full_path = os.path.join(nn_config.save_path_nn, filename)

        # Construct the "_last" filename
        last_filename = f"{nn_config.name}_last.txt"
        last_full_path = os.path.join(nn_config.save_path_nn, last_filename)

        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError as e:
                print(f"Error removing existing file {full_path}: {e}")

        # Serialize the network
        try:
            self._net.Serialize(full_path)
            if nn_config.verbose:
                print(f"Network serialized to: {full_path}")
            # Create the "_last" copy
            try:
                shutil.copy2(full_path, last_full_path)
                if nn_config.verbose:
                    print(f"Copied network state to: {last_full_path}")
            except OSError as e:
                print(f"Error copying {full_path} to {last_full_path}: {e}")

        except Exception as e:
            print(f"Error during network serialization to {full_path}: {e}")

    def _calculate_terminal_penalty(self, game_sim: GameLogic,
                                    steps_taken: int) -> float:
        """Calculates the terminal penalty/reward based on the final state.
           Lower score (penalty) is better. Aims to strongly reward landing
           and penalize passivity or failure states.
        """
        terminal_penalty = 0.0

        # Base penalty for steps taken (reduced to focus on outcome)
        terminal_penalty += steps_taken * 0.05

        # Large terminal rewards/penalties based on outcome
        if game_sim.landed_successfully:
            terminal_penalty -= 50000.0  # Significantly increased reward
            # Increased bonus for remaining fuel
            terminal_penalty -= game_sim.fuel * 5.0
        elif game_sim.crashed:
            terminal_penalty += 20000.0  # Increased penalty for crashing
            # Increased penalty based on final velocity magnitude
            final_v_mag = np.sqrt(game_sim.vx**2 + game_sim.vy**2)
            terminal_penalty += final_v_mag * 20.0
        elif game_sim.fuel <= 0 and not game_sim.landed:
            # Increased penalty for running out of fuel before landing
            terminal_penalty += 15000.0

        # Add penalties if not landed successfully (crash, no fuel, timeout)
        if not game_sim.landed_successfully:
            # Penalty for final distance to pad center
            final_dist_x = abs(game_sim.x - game_sim.landing_pad_center_x)
            final_dist_y = abs(game_sim.y - game_sim.landing_pad_y)
            final_dist_to_pad = np.sqrt(final_dist_x**2 + final_dist_y**2)
            # Significant penalty for being far away at the end
            terminal_penalty += final_dist_to_pad * 50.0

            # Penalty based on final velocity magnitude (kept)
            final_v_mag = np.sqrt(game_sim.vx**2 + game_sim.vy**2)
            terminal_penalty += final_v_mag * 10.0

        return terminal_penalty

    def train(self):
        if self._net is None:
            print("Error: Network not initialized or loaded.")
            return

        print(f"Starting training for {nn_config.epochs} generations...")
        print(f"Population size: {nn_config.population_size}, "
              f"Top performers: {nn_config.top_individuals}")
        print(f"Network structure: {self._nnsize}")
        if nn_config.multiple_layout:
            self.train_multiple_layout()
        else:
            self.train_single_layout()

    def train_multiple_layout(self):
        num_outputs = self._nnsize[-1]
        population_size = self._net.GetPopSize()
        num_layouts = nn_config.layout_nb

        print(f"Generating {num_layouts} layouts for training...")
        layouts = []
        left_to_right_count = 0
        right_to_left_count = 0
        for i in range(num_layouts):
            spad_x1, lpad_x1 = generate_random_pad_positions()
            layouts.append({'spad_x1': spad_x1, 'lpad_x1': lpad_x1})
            # Determine direction based on center points for accuracy
            spad_center = spad_x1 + game_cfg.spad_width / 2
            lpad_center = lpad_x1 + game_cfg.lpad_width / 2
            if spad_center < lpad_center:
                left_to_right_count += 1
            else:
                right_to_left_count += 1
            if nn_config.verbose:
                print(f"  Layout {i+1}: Start={spad_x1}, Landing={lpad_x1}")
        if nn_config.verbose:
            print(f"Initial layouts: {left_to_right_count} left-to-right, "
                  f"{right_to_left_count} right-to-left.")

        # --- Balance Layouts ---
        diff = abs(left_to_right_count - right_to_left_count)
        if diff > 1:
            num_to_flip = diff // 2
            # Use the imported cfg for screen width
            screen_center_x = cfg.width / 2.0
            if left_to_right_count > right_to_left_count:
                excess_direction = 'ltr'
                if nn_config.verbose:
                    print(f"  Balancing: Flipping {num_to_flip} left-to-right "
                          "layouts.")
            else:
                excess_direction = 'rtl'
                if nn_config.verbose:
                    print(f"  Balancing: Flipping {num_to_flip} right-to-left "
                          "layouts.")

            flipped_count = 0
            for i in range(num_layouts):
                if flipped_count >= num_to_flip:
                    break

                layout = layouts[i]
                spad_center = layout['spad_x1'] + game_cfg.spad_width / 2
                lpad_center = layout['lpad_x1'] + game_cfg.lpad_width / 2
                current_direction = 'ltr' if spad_center < lpad_center \
                    else 'rtl'

                if current_direction == excess_direction:
                    # Reflect positions around the screen center
                    old_spad_x1 = layout['spad_x1']
                    old_lpad_x1 = layout['lpad_x1']
                    # Calculate new top-left corner based on reflecting
                    old_spad_center = old_spad_x1 + game_cfg.spad_width / 2
                    old_lpad_center = old_lpad_x1 + game_cfg.lpad_width / 2
                    new_spad_center = 2 * screen_center_x - old_spad_center
                    new_lpad_center = 2 * screen_center_x - old_lpad_center
                    new_spad_x1 = new_spad_center - game_cfg.spad_width / 2
                    new_lpad_x1 = new_lpad_center - game_cfg.lpad_width / 2

                    # Ensure pads stay within bounds after flipping
                    # Use cfg.width for boundary check
                    new_spad_x1 = max(0, min(new_spad_x1,
                                             cfg.width - game_cfg.spad_width))
                    new_lpad_x1 = max(0, min(new_lpad_x1,
                                             cfg.width - game_cfg.lpad_width))

                    layouts[i]['spad_x1'] = new_spad_x1
                    layouts[i]['lpad_x1'] = new_lpad_x1

                    if nn_config.verbose:
                        print(f"    Flipped layout {i+1}: "
                              f"Old Start={old_spad_x1:.1f}, "
                              f"Old Landing={old_lpad_x1:.1f} -> "
                              f"New Start={new_spad_x1:.1f}, "
                              f"New Landing={new_lpad_x1:.1f}")

                    # Update counts
                    if excess_direction == 'ltr':
                        left_to_right_count -= 1
                        right_to_left_count += 1
                    else:
                        right_to_left_count -= 1
                        left_to_right_count += 1
                    flipped_count += 1

            print(f"Layouts: {left_to_right_count} left-to-right, "
                  f"{right_to_left_count} right-to-left.")
        # --- End Balance Layouts ---

        low_fitness_streak = 0  # Counter for consecutive low fit generations

        for gen in range(self._nGen, self._nGen + nn_config.epochs):
            print(f"\n--- Generation {gen + 1} ---")
            fitness_scores = np.zeros(population_size, dtype=np.float64)
            all_member_steps = []  # Track average steps per member

            for member_id in range(population_size):
                total_fitness_for_member = 0.0
                total_steps_for_member = 0

                for layout_idx, layout_info in enumerate(layouts):
                    game_sim = GameLogic(no_print=True)
                    # Set the specific layout for this run
                    set_pad_positions(layout_info['spad_x1'],
                                      layout_info['lpad_x1'])
                    # Important: Reset game state for the new layout
                    game_sim.reset()

                    state = game_sim.get_state()
                    done = False
                    accumulated_step_penalty = 0.0
                    steps = 0

                    while not done and steps < game_cfg.max_steps:
                        inputs = np.array(state, dtype=np.float64)
                        outputs = np.zeros(num_outputs, dtype=np.float64)

                        # Get action from NN for the current member
                        self._net.feedforward(inputs, outputs, member_id,
                                              False)
                        action = np.argmax(outputs)

                        # Update game state
                        next_state, done = game_sim.update(action)

                        # Calculate step penalties for this layout run
                        dist_x = abs(game_sim.x
                                     - game_sim.landing_pad_center_x)
                        dist_y = abs(game_sim.y
                                     - game_sim.landing_pad_y)
                        # Penalize slightly more for being
                        # far vertically near pad_y
                        y_penalty_factor = 1.0 + max(0, (
                            game_sim.landing_pad_y - game_sim.y) / 50.0)
                        # Increased step penalty weight to encourage progress
                        accumulated_step_penalty += (
                            dist_x * 0.5 + dist_y
                            * y_penalty_factor) * 0.01

                        state = next_state
                        steps += 1

                        if done:
                            break

                    # Calculate terminal penalty for this layout run
                    terminal_penalty = self._calculate_terminal_penalty(
                        game_sim, steps)
                    # Fitness for this single layout run
                    fitness_for_layout = accumulated_step_penalty + \
                        terminal_penalty

                    total_fitness_for_member += fitness_for_layout
                    total_steps_for_member += steps

                # Average fitness across all layouts for this member
                average_fitness = total_fitness_for_member / num_layouts
                fitness_scores[member_id] = average_fitness
                # Store average steps per layout for this member
                all_member_steps.append(total_steps_for_member / num_layouts)

            # Sort average fitness scores (ascending, lower is better)
            sorted_indices = np.argsort(fitness_scores)

            # Update NN weights and biases based on sorted average fitness
            self._net.UpdateWeightsAndBiases(sorted_indices)

            # Create the next population
            self._net.CreatePopulation(nn_config.elitism)

            # Update internal epoch counter in the C++ object
            self._net.UpdateEpochs(1)  # Increment epoch by 1
            self._nGen = self._net.GetEpochs()  # Sync python counter

            # Print generation summary using average values
            best_avg_fitness = fitness_scores[sorted_indices[0]]
            avg_fitness = np.mean(fitness_scores)
            # Calculate overall average steps across all members and layouts
            avg_steps = np.mean(all_member_steps)
            print(f"Generation {self._nGen} complete.")
            print(
                f"  Best Avg Fitness: {best_avg_fitness:.4f} (Member "
                f"{sorted_indices[0]})")
            print(f"  Avg Fitness:      {avg_fitness:.4f}")
            print(f"  Avg Steps/Layout: {avg_steps:.1f}")

            # --- Check for sustained low fitness (using best average fitness)
            if best_avg_fitness < nn_config.fit_min:
                low_fitness_streak += 1
                if nn_config.verbose:
                    print(f"  Low average fitness detected (Streak: "
                          f"{low_fitness_streak}).")
            else:
                if low_fitness_streak > 0 and nn_config.verbose:
                    print("  Average fitness recovered, "
                          "resetting low fitness streak.")
                low_fitness_streak = 0

            # If layouts needed to change, logic would go here.
            if low_fitness_streak > nn_config.fit_streak:
                print(f"--- Average fitness below {nn_config.fit_min} for "
                      f"{low_fitness_streak} generations. "
                      "Consider adjusting parameters or "
                      "layout generation. ---")
                # Action to take? Regenerate layouts? Stop?
                low_fitness_streak = 0  # Reset streak

            # Save network periodically
            if nn_config.save_nn and (
                    self._nGen % nn_config.save_interval == 0 or
                    gen == self._nGen + nn_config.epochs - 1):
                self.save()

        print("\nTraining finished.")

    def train_single_layout(self):
        num_outputs = self._nnsize[-1]
        population_size = self._net.GetPopSize()

        low_fitness_streak = 0  # Counter for consecutive low fit generations

        for gen in range(self._nGen, self._nGen + nn_config.epochs):
            print(f"\n--- Generation {gen + 1} ---")
            # Reset pad positions every nb_batches generations
            if gen % nn_config.nb_batches == 0:
                if nn_config.verbose:
                    print("--- Resetting pad positions for Generation "
                          f"{gen + 1} (Batch boundary) ---")
                reset_pad_positions()

            fitness_scores = np.zeros(population_size, dtype=np.float64)
            all_steps = []  # Track steps per member for info

            for member_id in range(population_size):
                game_sim = GameLogic(no_print=True)
                state = game_sim.get_state()
                done = False
                accumulated_step_penalty = 0.0  # Accumulate step penalties
                steps = 0

                while not done and steps < game_cfg.max_steps:
                    inputs = np.array(state, dtype=np.float64)
                    outputs = np.zeros(num_outputs, dtype=np.float64)

                    # Get action from NN
                    self._net.feedforward(inputs, outputs, member_id, False)
                    action = np.argmax(outputs)

                    # Update game state - returns (state, done)
                    next_state, done = game_sim.update(action)

                    # --- Calculate step penalties ---
                    # Penalty for distance from pad center
                    # (using state *after* update)
                    dist_x = abs(game_sim.x - game_sim.landing_pad_center_x)
                    dist_y = abs(game_sim.y - game_sim.landing_pad_y)
                    # Increased step penalty weight
                    accumulated_step_penalty += (dist_x + dist_y) * 0.005

                    state = next_state
                    steps += 1

                    if done:
                        break

                # Calculate terminal penalty based on final state
                terminal_penalty = self._calculate_terminal_penalty(game_sim,
                                                                    steps)
                # Final fitness is sum of accumulated step penalties
                # and terminal penalty
                fitness_scores[member_id] = accumulated_step_penalty + \
                    terminal_penalty
                all_steps.append(steps)

            # Sort fitness scores (ascending, lower is better) and get indices
            sorted_indices = np.argsort(fitness_scores)

            # Update NN weights and biases based on sorted fitness
            self._net.UpdateWeightsAndBiases(sorted_indices)

            # Create the next population
            self._net.CreatePopulation(nn_config.elitism)

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

            # --- Check for sustained low fitness and reset pads if needed ---
            if best_fitness < nn_config.fit_min:
                low_fitness_streak += 1
                if nn_config.verbose:
                    print(f"  Low fitness detected (Streak: "
                          f"{low_fitness_streak}).")
            else:
                if low_fitness_streak > 0 and nn_config.verbose:
                    print("  Fitness recovered, resetting low fitness streak.")
                low_fitness_streak = 0

            if low_fitness_streak > nn_config.fit_streak:
                print(f"--- Fitness below {nn_config.fit_min} for "
                      f"{low_fitness_streak} generations. "
                      "Resetting pad positions. ---")
                reset_pad_positions()
                low_fitness_streak = 0  # Reset streak after triggering reset
            # ----------------------------------------------------------------

            # Save network periodically
            if nn_config.save_nn and (
                    self._nGen % nn_config.save_interval == 0 or
                    gen == self._nGen + nn_config.epochs - 1):
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
    pass
