# Lunar Landing Simulation

A fully-playable Lunar Landing game in Python. Control the lander manually or enable an autoplay mode powered by a neural network. The simulation models lunar gravity and physics for a realistic landing experience, and supports AI auto-pilot through reinforcement learning.

## Features

- Manual control via keyboard,
- Neural network autoplay (reinforcement learning-based),
- Realistic physics simulation,
- Modular and extensible codebase.

https://github.com/user-attachments/assets/a9b05469-6b4c-4e42-b8c9-dafea75d5fee

## Requirements

- Git
- CMake
- clang
- Python

## Getting Started

To get started with the lunar lander:

1. Clone the repository:
   ```
   git clone https://github.com/azimonti/lunar-lander
   ```

2. Navigate to the repository directory:
   ```
   cd lunar-lander
   ```

3. Initialize and update the submodules:
  ```
  git submodule update --init --recursive
  ```

  Further update of the submodule can be done with the command:
  ```
  git submodule update --remote
  ```

4. Install required Python dependencies in a virtual environment and activate it:
   ```
   ./create_env.sh
   source "venv/bin/activate" #venv/Scripts/activate on MINGW
   ```

5. Compile the libraries and the game logic
  ```
  ./build_libs.sh
  ```

  If any error or missing dependencies for `ma-libs` please look at the instructions [here](https://github.com/azimonti/ma-libs)


6. Run the program
  ```
  python main.py --mode=play     # default - user plays the game
  ./externals/ma-libs/build/Release/main_train_d # optional --continue for training the neural network
  python main.py --mode=nn_play  # nn is playing the game using the last save neural network
  ```

A set of weights is available in `sample_runs`. It is possible to use a sample, copying the desired file to `data/lunar_lander_last.txt`.

## Bonus / Penalty Parameters

These parameters in `config.txt` configure per-step and episode-end reward and penalty shaping for neural network training in a landing environment. They control incentives for minimizing distance to target, efficient action selection, time and resource management, and penalize failure events such as crashing or fuel depletion. Tuning these values directly impacts learned agent behavior and policy optimality:

- `sp_dist_factor`: Multiplies the step-based penalty by current distance to pad,
- `sp_action_factor`: Multiplies the step-based penalty if an action is taken,
- `tp_steps_factor`: Scales penalty based on total episode steps,
- `tp_dist_factor`: Scales penalty based on final distance to the pad,
- `tp_landed_bonus`: Bonus for landing successfully,
- `tp_landed_lr_bonus`: Bonus for landing successfully on both direction,
- `tp_fuel_bonus_factor`: Scales bonus by fuel remaining after landing,
- `tp_crashed_penalty`: Penalty for crashing,
- `tp_crash_v_mag_factor`: Additional penalty scaled by crash velocity,
- `tp_no_fuel_penalty`: Penalty for running out of fuel before landing.

Example:

```python
# Bonus / Penalties
# Each Step
NNTrainingCfg.sp_dist_factor = 0.001                                     # Step based on distance to pad
NNTrainingCfg.sp_action_factor = 0.01                                    # Step if an action is taken
# Final
NNTrainingCfg.tp_steps_factor = 0.1                                      # Total steps taken
NNTrainingCfg.tp_dist_factor = 0.5                                       # Final distance to pad
NNTrainingCfg.tp_landed_bonus = 0.0                                      # Bonus for successful landing
NNTrainingCfg.tp_landed_lr_bonus = 10000.0                               # Bonus for successful combined landing
NNTrainingCfg.tp_fuel_bonus_factor = 2.0                                 # Bonus factor for remaining fuel on landing
NNTrainingCfg.tp_crashed_penalty = 500.0                                 # Crashing
NNTrainingCfg.tp_crash_v_mag_factor = 10.0                               # Velocity magnitude on crash
NNTrainingCfg.tp_no_fuel_penalty = 500.0                                 # Running out of fuel without landing
```

The most effective configuration for final rewards uses only a small penalty / bonus for fuel consumption to encourage efficiency, a penalty proportional to crash velocity to enforce soft landings, and a large bonus for combined landings to strongly incentivize proper bidirectional maneuvers and stable touchdown. Other reward or penalty terms are set to zero.

```python
# Final
NNTrainingCfg.tp_landed_bonus = 0.0                                      # Bonus for successful landing
NNTrainingCfg.tp_landed_lr_bonus = 10000.0                               # Bonus for successful combined landing
NNTrainingCfg.tp_fuel_bonus_factor = 2.0                                 # Bonus factor for remaining fuel on landing
NNTrainingCfg.tp_crashed_penalty = 0.0                                   # Crashing
NNTrainingCfg.tp_crash_v_mag_factor = 10.0                               # Velocity magnitude on crash
NNTrainingCfg.tp_no_fuel_penalty = 10.0                                  # Running out of fuel without landing
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or want to get in touch regarding the project, please open an issue or contact the repository maintainers directly through GitHub.
