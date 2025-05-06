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
import time
import math

from mod_config import cfg, nn_config, game_cfg, planet_cfg, lander_cfg, \
    reset_pad_positions, generate_random_pad_positions, set_pad_positions

cwd = os.getcwd()
build_dir = os.path.join(cwd, "./externals/ma-libs/build")
if "DEBUG" in os.environ:
    build_path = os.path.join(build_dir, "Debug")
else:
    build_path = os.path.join(build_dir, "Release")
sys.path.append(os.path.realpath(build_path))
try:
    import cpp_nn_py2 as cpp_nn_py
    import cpp_game_logic
except ModuleNotFoundError as e:
    print(f"Error: {e}")


class NeuralNetwork():
    def __init__(self):
        self._net = None
        self._nGen = 0
        self._nnsize = None
        self._game_env = None
        self._start_time = time.time()

    def init(self):
        if nn_config.save_nn:
            # create directory if it doesn't exist
            os.makedirs(nn_config.save_path_nn, exist_ok=True)
        # GameLogic expects 5 inputs and 4 outputs
        self._nnsize = [5] + nn_config.hlayers + [4]

        if nn_config.use_float:
            self._net = cpp_nn_py.ANN_MLP_GA_float(
                self._nnsize, nn_config.seed, nn_config.population_size,
                nn_config.top_individuals, nn_config.activation_id,
                nn_config.elitism)
        else:
            self._net = cpp_nn_py.ANN_MLP_GA_double(
                self._nnsize, nn_config.seed, nn_config.population_size,
                nn_config.top_individuals, nn_config.activation_id,
                nn_config.elitism)

        self._net.SetName(nn_config.name)
        self._net.SetMixed(nn_config.mixed_population)
        self._net.CreatePopulation(True)
        self._nGen = 0

    def load(self, step='last'):
        if nn_config.use_float:
            self._net = cpp_nn_py.ANN_MLP_GA_float()
        else:
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
        self._start_time = time.time()  # Reset start time on load
        self._nnsize = self._net.GetNetworkSize()
        if nn_config.verbose:
            print(f"Loading network from: {full_path} "
                  f"(Generation {self._nGen})")
            print(f"  Expected network structure: {self._nnsize}")

    def save(self):
        """Saves the network state and creates a _last copy."""

        base_prefix = nn_config.name

        if not nn_config.overwrite:
            layout_type = 'm' if nn_config.multiple_layout else 's'
            nn_size_str = "_".join(map(str, self._nnsize))
            # Format step number with leading zeros
            step_num_str = f"{self._nGen:04d}"
            filename = (f"{base_prefix}_{layout_type}_{nn_size_str}_s"
                        f"_{step_num_str}.txt")
        else:
            # If overwrite is true, maybe use a simpler name or the last format
            # Or adjust as needed
            filename = f"{base_prefix}_latest_overwrite.txt"

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

    def _calculate_step_penalty(self, game_sim: cpp_game_logic.GameLogicCpp,
                                action: int) -> float:
        """Calculates the penalty applied at each step. Lower score is better.
        """
        # No longer used - moved to C++
        step_penalty = 0.0

        # Penalty for distance from pad center
        dist_x = abs(game_sim.x - game_sim.landing_pad_center_x)
        dist_y = abs(game_sim.y - game_sim.landing_pad_y)
        step_penalty += (dist_x + dist_y) * 0.001

        # Optional: Penalty for using fuel (if action > 0)
        if action > 0:
            step_penalty += 0.01  # Small penalty per thrust action

        return step_penalty

    def _calculate_terminal_penalty(self,
                                    game_sim: cpp_game_logic.GameLogicCpp,
                                    steps_taken: int) -> float:
        """Calculates the terminal penalty/reward based on the final state.
           Lower score (penalty) is better.
        """
        # No longer used - moved to C++
        terminal_penalty = 0.0

        # Base penalty for steps taken (encourages efficiency)
        terminal_penalty += steps_taken * 0.1

        # Distance penalty (applied more heavily at the end)
        dist_x = abs(game_sim.x - game_sim.landing_pad_center_x)
        dist_y = abs(game_sim.y - game_sim.landing_pad_y)
        # Scale distance penalty - higher if further away
        terminal_penalty += (dist_x + dist_y) * 0.5

        if game_sim.landed_successfully:
            terminal_penalty -= 1000.0  # Big reward (negative penalty)
            # Bonus for remaining fuel
            terminal_penalty -= game_sim.fuel * 2.0
        elif game_sim.crashed:
            terminal_penalty += 500.0  # Penalty for crashing
            # Increase penalty based on final velocity magnitude
            final_v_mag = np.sqrt(game_sim.vx**2 + game_sim.vy**2)
            terminal_penalty += final_v_mag * 10.0
        elif game_sim.fuel <= 0 and not game_sim.landed:
            terminal_penalty += 250.0  # Penalty for running out of fuel in air

        # Add small penalty based on final velocity if not landed successfully
        if not game_sim.landed_successfully:
            final_v_mag = np.sqrt(game_sim.vx**2 + game_sim.vy**2)
            terminal_penalty += final_v_mag * 1.0

        return terminal_penalty

    def _format_time(self, seconds):
        """Formats seconds into mm:ss string."""
        if seconds < 0:
            return "00:00"
        minutes = math.floor(seconds / 60)
        remaining_seconds = math.floor(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def train(self):
        if self._net is None:
            print("Error: Network not initialized or loaded.")
            return

        print(f"Starting training for {nn_config.epochs} generations...")
        print(f"Population size: {self._net.GetPopSize()}, "
              f"Top performers: {self._net.GetTopPerformersSize()}")
        print(f"Network structure: {self._net.GetNetworkSize()}")
        if nn_config.multiple_layout:
            self.train_multiple_layout()
        else:
            self.train_single_layout()

    def train_multiple_layout(self):
        population_size = self._net.GetPopSize()
        num_layouts = nn_config.layout_nb

        if nn_config.verbose:
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

        # --- Balance Layouts based on Ratio ---
        ratio = nn_config.left_right_ratio
        # Validate ratio
        if not (0.0 <= ratio <= 1.0):
            print(f"  Warning: Invalid left_right_ratio ({ratio}). "
                  "Defaulting to 0.5.")
            ratio = 0.5

        target_ltr_count = round(num_layouts * ratio)
        current_ltr_count = left_to_right_count
        num_to_flip = abs(current_ltr_count - target_ltr_count)

        if num_to_flip > 0:
            screen_center_x = cfg.width / 2.0
            if current_ltr_count > target_ltr_count:
                # Need to flip LTR to RTL
                direction_to_find = 'ltr'
                if nn_config.verbose:
                    print(f"  Balancing (Ratio {ratio}): Flipping "
                          f"{num_to_flip} left-to-right layouts to "
                          f"reach target {target_ltr_count}.")
            else:
                # Need to flip RTL to LTR
                direction_to_find = 'rtl'
                if nn_config.verbose:
                    print(f"  Balancing (Ratio {ratio}): Flipping "
                          f"{num_to_flip} right-to-left layouts to "
                          f"reach target {target_ltr_count}.")

            flipped_count = 0
            # Iterate through layouts to find candidates for flipping
            for i in range(num_layouts):
                if flipped_count >= num_to_flip:
                    break  # Flipped enough layouts

                layout = layouts[i]
                spad_center = layout['spad_x1'] + game_cfg.spad_width / 2
                lpad_center = layout['lpad_x1'] + game_cfg.lpad_width / 2
                current_direction = 'ltr' if spad_center < lpad_center \
                    else 'rtl'

                if current_direction == direction_to_find:
                    # This layout is a candidate for flipping
                    old_spad_x1 = layout['spad_x1']
                    old_lpad_x1 = layout['lpad_x1']

                    # Reflect positions around the screen center
                    old_spad_center = old_spad_x1 + game_cfg.spad_width / 2
                    old_lpad_center = old_lpad_x1 + game_cfg.lpad_width / 2
                    new_spad_center = 2 * screen_center_x - old_spad_center
                    new_lpad_center = 2 * screen_center_x - old_lpad_center
                    new_spad_x1 = new_spad_center - game_cfg.spad_width / 2
                    new_lpad_x1 = new_lpad_center - game_cfg.lpad_width / 2

                    # Ensure pads stay within bounds after flipping
                    new_spad_x1 = max(0, min(new_spad_x1,
                                             cfg.width - game_cfg.spad_width))
                    new_lpad_x1 = max(0, min(new_lpad_x1,
                                             cfg.width - game_cfg.lpad_width))

                    # Apply the flip
                    layouts[i]['spad_x1'] = new_spad_x1
                    layouts[i]['lpad_x1'] = new_lpad_x1

                    if nn_config.verbose:
                        print(f"    Flipped layout {i+1}: "
                              f"Old Start={old_spad_x1:.1f}, "
                              f"Old Landing={old_lpad_x1:.1f} -> "
                              f"New Start={new_spad_x1:.1f}, "
                              f"New Landing={new_lpad_x1:.1f}")

                    # Update counts immediately after flipping
                    if direction_to_find == 'ltr':
                        left_to_right_count -= 1
                        right_to_left_count += 1
                    else:  # direction_to_find == 'rtl'
                        right_to_left_count -= 1
                        left_to_right_count += 1
                    flipped_count += 1

            # Recalculate current_ltr_count after flips for final report
            current_ltr_count = left_to_right_count
            if nn_config.verbose:
                print(f"Layouts after ratio balancing: {current_ltr_count} "
                      "left-to-right "
                      f"({current_ltr_count/num_layouts*100:.1f}%), "
                      f"{right_to_left_count} right-to-left.")
        else:
            if nn_config.verbose:
                print(f"Layouts already meet target ratio ({ratio}): "
                      f"{left_to_right_count} left-to-right, "
                      f"{right_to_left_count} right-to-left.")
        print(f"Layouts: {left_to_right_count} left-to-right, "
              f"{right_to_left_count} right-to-left.")
        # --- End Balance Layouts ---

        for gen in range(self._nGen, self._nGen + nn_config.epochs):
            gen_start_time = time.time()
            print(f"\n--- Generation {gen + 1} ---")
            fitness_scores = np.zeros(population_size, dtype=np.float64)
            all_member_steps = []  # Track average steps per member

            for member_id in range(population_size):
                total_fitness_for_member = 0.0
                total_steps_for_member = 0

                for layout_idx, layout_info in enumerate(layouts):
                    # Use the C++ GameLogic implementation
                    game_sim = cpp_game_logic.GameLogicCpp(no_print_flag=True)
                    set_pad_positions(layout_info['spad_x1'],
                                      layout_info['lpad_x1'])

                    # --- Configure the C++ Game Logic Instance ---
                    # Use the current global config values
                    game_sim.set_config(
                        cfg_w=cfg.width,
                        cfg_h=cfg.height, gcfg_pad_y1=game_cfg.pad_y1,
                        gcfg_terrain_y_val=game_cfg.terrain_y,
                        gcfg_max_v_x=game_cfg.max_vx,
                        gcfg_max_v_y=game_cfg.max_vy,
                        pcfg_gravity=planet_cfg.g,
                        pcfg_fric_x=planet_cfg.mu_x,
                        pcfg_fric_y=planet_cfg.mu_y,
                        lcfg_w=lander_cfg.width, lcfg_h=lander_cfg.height,
                        lcfg_fuel=lander_cfg.max_fuel,
                        gcfg_spad_width=game_cfg.spad_width,
                        gcfg_lpad_width=game_cfg.lpad_width,
                        gcfg_x0_vec=game_cfg.x0.tolist(),
                        gcfg_v0_vec=game_cfg.v0.tolist(),
                        gcfg_a0_vec=game_cfg.a0.tolist()
                    )

                    # --- Reset the C++ logic using the CURRENT pad pos  ---
                    # Ensures the instance uses the specific layout for the run
                    game_sim.reset(game_cfg.spad_x1, game_cfg.lpad_x1)

                    # Pre-allocate state buffer
                    state_buffer = np.zeros(5, dtype=np.float64)
                    game_sim.get_state(state_buffer) # Get initial state into buffer
                    done = False
                    accumulated_step_penalty = 0.0
                    steps = 0

                    # Determine dtype based on config
                    dtype = np.float32 if nn_config.use_float else np.float64

                    while not done and steps < game_cfg.max_steps:
                        inputs = np.array(state_buffer, dtype=dtype) # Use buffer
                        # Get action index directly from NN
                        action = self._net.feedforwardIndex(inputs, member_id)
                        # Update game state - writes state into buffer, returns done
                        done = game_sim.update(action, state_buffer)
                        # Calculate step penalty using C++ method
                        accumulated_step_penalty += \
                            game_sim.calculate_step_penalty(action)
                        # state = next_state # No longer needed, buffer updated in-place
                        steps += 1
                        if done:
                            break

                    # Calculate terminal penalty using C++ method
                    terminal_penalty = game_sim.calculate_terminal_penalty(
                        steps)
                    # Final fitness
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
            # Calculate times
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            total_elapsed_time = gen_end_time - self._start_time
            # Print generation summary using average values
            best_avg_fitness = fitness_scores[sorted_indices[0]]
            avg_fitness = np.mean(fitness_scores)
            # Calculate overall average steps across all members and layouts
            avg_steps = np.mean(all_member_steps)
            print(f"Generation {self._nGen} complete. "
                  f"(Took: {self._format_time(gen_duration)}, "
                  f"Total: {self._format_time(total_elapsed_time)})")
            print(
                f"  Best Avg Fitness: {best_avg_fitness:.4f} (Member "
                f"{sorted_indices[0]})")
            print(f"  Avg Fitness:      {avg_fitness:.4f}")
            print(f"  Avg Steps/Layout: {avg_steps:.1f}")

            # Save network periodically
            if nn_config.save_nn and (
                    self._nGen % nn_config.save_interval == 0 or
                    gen == self._nGen + nn_config.epochs - 1):
                self.save()

        print("\nTraining finished.")

    def train_single_layout(self):
        population_size = self._net.GetPopSize()

        for gen in range(self._nGen, self._nGen + nn_config.epochs):
            gen_start_time = time.time()
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
                # Use the C++ GameLogic implementation
                game_sim = cpp_game_logic.GameLogicCpp(no_print_flag=True)

                # --- Configure the C++ Game Logic Instance ---
                # Use the current global config values
                game_sim.set_config(
                    cfg_w=cfg.width,
                    cfg_h=cfg.height, gcfg_pad_y1=game_cfg.pad_y1,
                    gcfg_terrain_y_val=game_cfg.terrain_y,
                    gcfg_max_v_x=game_cfg.max_vx,
                    gcfg_max_v_y=game_cfg.max_vy,
                    pcfg_gravity=planet_cfg.g,
                    pcfg_fric_x=planet_cfg.mu_x,
                    pcfg_fric_y=planet_cfg.mu_y,
                    lcfg_w=lander_cfg.width, lcfg_h=lander_cfg.height,
                    lcfg_fuel=lander_cfg.max_fuel,
                    gcfg_spad_width=game_cfg.spad_width,
                    gcfg_lpad_width=game_cfg.lpad_width,
                    gcfg_x0_vec=game_cfg.x0.tolist(),
                    gcfg_v0_vec=game_cfg.v0.tolist(),
                    gcfg_a0_vec=game_cfg.a0.tolist()
                )

                # --- Reset the C++ logic using the CURRENT pad positions ---
                # Ensures the instance uses the potentially randomized pads
                game_sim.reset(game_cfg.spad_x1, game_cfg.lpad_x1)

                # Pre-allocate state buffer
                state_buffer = np.zeros(5, dtype=np.float64)
                game_sim.get_state(state_buffer) # Get initial state into buffer
                done = False
                accumulated_step_penalty = 0.0
                steps = 0
                # Determine dtype based on config
                dtype = np.float32 if nn_config.use_float else np.float64

                while not done and steps < game_cfg.max_steps:
                    inputs = np.array(state_buffer, dtype=dtype) # Use buffer
                    # Get action index directly from NN
                    action = self._net.feedforwardIndex(inputs, member_id)
                    # Update game state - writes state into buffer, returns done
                    done = game_sim.update(action, state_buffer)
                    # Calculate step penalty using C++ method
                    accumulated_step_penalty += \
                        game_sim.calculate_step_penalty(action)
                    steps += 1
                    if done:
                        break
                # Calculate terminal penalty using C++ method
                terminal_penalty = game_sim.calculate_terminal_penalty(steps)
                # Final fitness
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
            # Calculate times
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            total_elapsed_time = gen_end_time - self._start_time
            # Print generation summary
            best_fitness = fitness_scores[sorted_indices[0]]
            avg_fitness = np.mean(fitness_scores)
            avg_steps = np.mean(all_steps)
            print(f"Generation {self._nGen} complete. "
                  f"(Took: {self._format_time(gen_duration)}, "
                  f"Total: {self._format_time(total_elapsed_time)})")
            print(
                f"  Best Fitness: {best_fitness:.4f} (Member "
                f"{sorted_indices[0]})")
            print(f"  Avg Fitness:  {avg_fitness:.4f}")
            print(f"  Avg Steps:    {avg_steps:.1f}")

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

        # Determine dtype based on config
        dtype = np.float32 if nn_config.use_float else np.float64
        inputs = np.array(current_state, dtype=dtype)

        # Use member_id 0 - in the loaded network represents the best
        # or the result of the population evolution.
        # The GA manages this internally.
        try:
            return self._net.feedforwardIndex(inputs, 0)
        except Exception as e:
            print(f"Error during feedforward: {e}")
            return 0


if __name__ == '__main__':
    pass
