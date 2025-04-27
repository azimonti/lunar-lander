'''
/*********************/
/*  game_logic.py    */
/*   Version 1.0     */
/*    2025/04/27     */
/*********************/
'''
import numpy as np
from mod_config import cfg, game_cfg as gcfg, planet_cfg as pcfg, \
    lander_cfg as lcfg


class GameLogic:
    def __init__(self, no_print=False):
        self.reset()
        # Constants from config
        self.landing_pad_center_x = gcfg.lpad_x1 + gcfg.lpad_width / 2
        self.landing_pad_y = gcfg.pad_y1  # Using top boundary as reference
        self.no_print = no_print

    def p(self, text):
        if self.no_print:
            return
        print(text)

    def reset(self):
        """Resets the game state to the initial configuration."""
        self.x = gcfg.x0[0]
        self.y = gcfg.x0[1]
        self.vx = gcfg.v0[0]
        self.vy = gcfg.v0[1]
        # Initial acceleration includes gravity
        self.ax = gcfg.a0[0]
        self.ay = gcfg.a0[1] + pcfg.g  # Gravity is always acting
        self.fuel = float(lcfg.max_fuel)
        self.landed = False
        self.crashed = False
        self.landed_successfully = False
        self.out_of_bounds = False
        self.time_step = 0
        self.action_log = []  # To track actions if needed

    def _apply_action(self, action: int):
        """Applies the chosen action (thrust).
        0: Noop, 1: Up, 2: Left, 3: Right"""
        self.action_log.append(action)  # Log action
        if self.fuel <= 0:
            return  # No thrust if out of fuel

        if action == 1:  # Thrust Up
            # Apply vertical thrust impulse (counteracting gravity)
            self.vy -= 0.3  # Instantaneous change, adjust as needed
            self.fuel -= 1
        elif action == 2:  # Thrust Left
            self.vx -= 0.05
            self.fuel -= 0.5
        elif action == 3:  # Thrust Right
            self.vx += 0.05
            self.fuel -= 0.5
        # Action 0 is Noop, do nothing

        # Clamp fuel to zero
        self.fuel = max(0, self.fuel)

    def _update_physics(self):
        """Updates the lander's position and velocity based on physics."""
        # Update velocity with acceleration (gravity is already in ay)
        # self.vx += self.ax # No horizontal acceleration
        self.vy += self.ay

        # Apply friction
        self.vx *= (1 - pcfg.mu_x)
        self.vy *= (1 - pcfg.mu_y)  # Air resistance/friction

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Check boundaries (simple check for now)
        if self.x < 0 or self.x + lcfg.width > cfg.width:
            self.vx *= -0.5  # Bounce off sides slightly
            self.x = np.clip(self.x, 0, cfg.width - lcfg.width)
            # self.crashed = True # Or just bounce?
            # self.out_of_bounds = True

        # Prevent going below ground level before landing check
        ground_level = cfg.height - lcfg.height - gcfg.terrain_y
        if self.y > ground_level:
            self.y = ground_level
            # Don't zero velocity here yet, check landing conditions first

    def _check_landing_crash(self):
        """Checks if the lander has landed or crashed.
        Only performs check after first step."""
        # Only check for landing/crash after the first time step
        # to avoid immediate crash on start
        if self.time_step == 0:
            return

        ground_level = cfg.height - lcfg.height - gcfg.terrain_y
        if self.y >= ground_level:
            self.y = ground_level  # Ensure it rests on the ground
            self.landed = True

            on_landing_pad = (self.x >= gcfg.lpad_x1 and
                              self.x + lcfg.width <= (gcfg.lpad_x1
                                                      + gcfg.lpad_width))
            safe_speed = (abs(self.vx) < abs(gcfg.max_vx) and
                          abs(self.vy) < abs(gcfg.max_vy))

            if on_landing_pad and safe_speed:
                self.landed_successfully = True
                self.vx = 0  # Stop movement on successful landing
                self.vy = 0
                self.p("GameLogic: Successful landing!")
            else:
                self.crashed = True
                self.landed_successfully = False
                # Stop movement on crash
                self.vx = 0
                self.vy = 0
                if not on_landing_pad:
                    self.p("GameLogic: Crashed - outside landing area!")
                if abs(self.vx) >= abs(gcfg.max_vx):
                    self.p("GameLogic: Crashed - excessive horizontal speed! "
                           f"(|{self.vx:.2f}| >= {gcfg.max_vx})")
                if abs(self.vy) >= abs(gcfg.max_vy):
                    self.p("GameLogic: Crashed - excessive vertical speed! ("
                           f"|{self.vy:.2f}| >= {gcfg.max_vy})")

    def update(self, action: int):
        """Performs one time step of the game simulation.
           Args: action (int): The action to apply
                               (0: Noop, 1: Up, 2: Left, 3: Right).
           Returns: tuple: (state, reward, done)
                      state (np.array): The current state vector for the NN.
                      reward (float): The reward obtained in this step.
                      done (bool): Whether the episode has ended.
        """
        if self.is_done():
            return self.get_state(), 0.0, True

        self._apply_action(action)
        self._update_physics()
        self._check_landing_crash()

        self.time_step += 1

        done = self.is_done()
        reward = self._calculate_reward(done)
        state = self.get_state()

        return state, reward, done

    def is_done(self) -> bool:
        """Checks if the game episode has finished."""
        # return self.landed or self.crashed or self.fuel <= 0
        # or self.out_of_bounds
        # For now, it ends only on landing/crash or add a time limit later
        return self.landed or self.crashed

    def _calculate_reward(self, done: bool) -> float:
        """Calculates the reward for the current state/action.
           Placeholder implementation. Needs refinement for RL.
        """
        reward = 0.0

        # Penalty for fuel consumption
        # reward -= 0.1 # Small penalty per step?
        # reward -= (self.action_log[-1] > 0) * 0.05 # Penalty for using fuel

        # Shaping reward: encourage getting closer to the pad
        dist_x = self.x - self.landing_pad_center_x
        dist_y = self.y - self.landing_pad_y  # Target is above ground
        reward -= (abs(dist_x) + abs(dist_y)) * 0.001  # Penalize distance

        if done:
            if self.landed_successfully:
                reward += 100.0  # Large reward for success
                reward -= self.time_step * 0.1  # Penalize taking too long
            elif self.crashed:
                reward -= 100.0  # Large penalty for crashing
            # elif self.fuel <= 0 and not self.landed:
            #     reward -= 50.0 # Penalty for running out of fuel mid-air
        return reward

    def get_state(self) -> np.ndarray:
        """Returns the current state vector for the NN."""
        # Calculate distance to target landing pad center
        dist_target_x = self.x - self.landing_pad_center_x
        dist_target_y = self.y - self.landing_pad_y  # Target Y is pad top

        # Normalize? For now, use raw values.
        # Normalization depends on the NN architecture and training process.
        state = np.array([
            self.x / cfg.width,             # Normalized X position
            self.y / cfg.height,            # Normalized Y position
            # Normalized Vx (relative to max safe speed)
            self.vx / gcfg.max_vx,
            # Normalized Vy (relative to max safe speed)
            self.vy / gcfg.max_vy,
            self.fuel / lcfg.max_fuel,      # Normalized Fuel
            dist_target_x / cfg.width,      # Normalized distance X
            dist_target_y / cfg.height,     # Normalized distance Y
            # self.time_step / MAX_STEPS ? # Optional: Normalized
        ], dtype=float)
        # Ensure vx/vy normalization doesn't cause issues
        # if max speeds are very small or zero
        # Clip normalized velocity to avoid extremes
        state[2] = np.clip(state[2], -5, 5)
        state[3] = np.clip(state[3], -5, 5)

        return state

    def get_render_info(self) -> dict:
        """Returns information needed for rendering."""
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "fuel": self.fuel,
            "last_action": self.action_log[-1] if self.action_log else 0,
            "landed": self.landed,
            "crashed": self.crashed,
            "landed_successfully": self.landed_successfully
        }


if __name__ == '__main__':
    # Example usage (optional)
    logic = GameLogic()
    state = logic.get_state()
    print("Initial State:", state)
    done = False
    step = 0
    total_reward = 0
    # Simulate a few steps with random actions
    while not done and step < 500:
        action = np.random.randint(0, 4)  # Random action
        state, reward, done = logic.update(action)
        total_reward += reward
        # print(f"Step {step}: Action={action}, Reward={reward:.2f}, "
        #      f"Done={done}")
        # print(f" State: x={logic.x:.1f}, y={logic.y:.1f}, "
        #      f"vx={logic.vx:.2f}, vy={logic.vy:.2f}, "
        #      f"fuel={logic.fuel:.1f}")
        step += 1

    print(f"\nSimulation finished after {step} steps.")
    print(f"Final State: x={logic.x:.1f}, y={logic.y:.1f}, "
          f"vx={logic.vx:.2f}, vy={logic.vy:.2f}, fuel={logic.fuel:.1f}")
    print(f"Landed: {logic.landed}, Crashed: {logic.crashed}, "
          f"Success: {logic.landed_successfully}")
    print(f"Total Reward: {total_reward:.2f}")
    print("Final State Vector:", state)
