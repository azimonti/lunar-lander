'''
/*******************/
/* mod_nn_train.py */
/*  Version 2.0    */
/*   2025/05/06    */
/*******************/
'''
import numpy as np
import os
import sys
import shutil
import time
import math

from mod_config import cfg, nn_config, game_cfg, planet_cfg, lander_cfg

cwd = os.getcwd()
build_dir = os.path.join(cwd, "./externals/ma-libs/build")
if "DEBUG" in os.environ:
    build_path = os.path.join(build_dir, "Debug")
else:
    build_path = os.path.join(build_dir, "Release")
sys.path.append(os.path.realpath(build_path))
try:
    import cpp_nn_py2 as cpp_nn_py
    from lunar_lander_cpp import GameLogicCppDouble as GameLogicCpp, \
            generate_random_pad_positions
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
            nn_size_str = "_".join(map(str, self._nnsize))
            # Format step number with leading zeros
            step_num_str = f"{self._nGen:04d}"
            filename = (f"{base_prefix}_{nn_size_str}_s_{step_num_str}.txt")
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

    def _format_time(self, seconds):
        """Formats seconds into mm:ss string."""
        if seconds < 0:
            return "00:00"
        minutes = math.floor(seconds / 60)
        remaining_seconds = math.floor(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def train_generation_f32(self, state_buffer, layouts, population_size,
                             fitness_scores, all_member_steps):
        inputs = np.zeros(len(state_buffer), dtype=np.float32)
        for member_id in range(population_size):
            total_fitness_for_member = 0.0
            total_steps_for_member = 0

            for layout_idx, layout_info in enumerate(layouts):
                # Use the C++ GameLogic implementation
                game_sim = GameLogicCpp(no_print_flag=True)

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
                game_sim.reset(layout_info['spad_x1'], layout_info['lpad_x1'])

                # Get initial state into buffer
                game_sim.get_state(state_buffer)
                done = False
                accumulated_step_penalty = 0.0
                steps = 0

                while steps < game_cfg.max_steps:
                    inputs = state_buffer
                    # Get action index directly from NN
                    action = self._net.feedforwardIndex(inputs, member_id)
                    # Update game state - writes state into buffer,
                    # returns done
                    done = game_sim.update(action, state_buffer)
                    # Calculate step penalty using C++ method
                    accumulated_step_penalty += \
                        game_sim.calculate_step_penalty(action)
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

            fitness_scores[member_id] = total_fitness_for_member
            # Store average steps per layout for this member
            all_member_steps[member_id] = total_steps_for_member

    def train_generation_d64(self, state_buffer, layouts, population_size,
                             fitness_scores, all_member_steps):
        for member_id in range(population_size):
            total_fitness_for_member = 0.0
            total_steps_for_member = 0

            for layout_idx, layout_info in enumerate(layouts):
                # Use the C++ GameLogic implementation
                game_sim = GameLogicCpp(no_print_flag=True)

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
                game_sim.reset(layout_info['spad_x1'], layout_info['lpad_x1'])

                # Get initial state into buffer
                game_sim.get_state(state_buffer)
                done = False
                accumulated_step_penalty = 0.0
                steps = 0

                while steps < game_cfg.max_steps:
                    action = self._net.feedforwardIndex(state_buffer,
                                                        member_id)
                    # Update game state - writes state into buffer,
                    # returns done
                    done = game_sim.update(action, state_buffer)
                    # Calculate step penalty using C++ method
                    accumulated_step_penalty += \
                        game_sim.calculate_step_penalty(action)
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

            fitness_scores[member_id] = total_fitness_for_member
            # Store average steps per layout for this member
            all_member_steps[member_id] = total_steps_for_member

    def train(self):
        if self._net is None:
            print("Error: Network not initialized or loaded.")
            return

        print(f"Starting training for {nn_config.epochs} generations...")
        print(f"Population size: {self._net.GetPopSize()}, "
              f"Top performers: {self._net.GetTopPerformersSize()}")
        print(f"Network structure: {self._net.GetNetworkSize()}")
        num_layouts = nn_config.layout_nb

        if nn_config.verbose:
            print(f"Generating {num_layouts} layouts for training...")
        layouts = []
        left_to_right_count = 0
        right_to_left_count = 0
        for i in range(num_layouts):
            spad_x1, lpad_x1 = generate_random_pad_positions(
                cfg.width, game_cfg.spad_width, game_cfg.lpad_width)
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

        # Split in two function give at least a 5% improvements in speed
        # avoid checks when not needed
        train_generation_f = self.train_generation_f32 if \
            nn_config.use_float \
            else self.train_generation_d64

        # Pre-allocate memory
        population_size = self._net.GetPopSize()

        fitness_scores = np.zeros(population_size, dtype=np.float64)
        all_member_steps = np.zeros(population_size, dtype=np.float64)
        state_buffer = np.zeros(5, dtype=np.float64)

        for gen in range(self._nGen, self._nGen + nn_config.epochs):
            gen_start_time = time.time()
            print(f"\n--- Generation {gen + 1} ---")
            train_generation_f(state_buffer, layouts, population_size,
                               fitness_scores, all_member_steps)
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
            best_avg_fitness = fitness_scores[sorted_indices[0]] / num_layouts
            avg_fitness = np.mean(fitness_scores)
            avg_fitness /= num_layouts
            # Calculate overall average steps across all members and layouts
            avg_steps = np.mean(all_member_steps) / num_layouts
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
