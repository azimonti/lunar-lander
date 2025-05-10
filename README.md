# Lunar Landing Simulation

A fully-playable Lunar Landing game in Python. Control the lander manually or enable an autoplay mode powered by a neural network. The simulation models lunar gravity and physics for a realistic landing experience, and supports AI auto-pilot through reinforcement learning.

## Features

- Manual control via keyboard,
- Neural network autoplay (reinforcement learning-based),
- Realistic physics simulation,
- Modular and extensible codebase.

https://github.com/user-attachments/assets/a9feb50e-0872-41a3-84ee-86dfc4e0c948

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or want to get in touch regarding the project, please open an issue or contact the repository maintainers directly through GitHub.
