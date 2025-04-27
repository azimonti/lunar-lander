# Lunar Landing Simulation

A fully-playable Lunar Landing game in Python. Control the lander manually or enable an autoplay mode powered by a neural network. The simulation models lunar gravity and physics for a realistic landing experience, and supports AI auto-pilot through reinforcement learning.

## Features

- Manual control via keyboard,
- Neural network autoplay (reinforcement learning-based),
- Realistic physics simulation,
- Modular and extensible codebase.

## Requirements

- Git
- CMake
- clang
- Python

## Getting Started

To get started with the control feedback:

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

4. Install required Python dependencies (a virtual environment can be created running the script `create_env.sh`):
   ```
   pip install -r requirements.txt
   ```

5. Compile the libraries in `ma-libs`
  ```
  cd externals/ma-libs
  # optional steps if dependencies are not installed globally
  # ./manage_dependency_libraries.sh -d
  # ./manage_dependency_libraries.sh -b
  ./cbuild.sh --build-type Debug --cmake-params "-DCPP_LIBNN=ON -DCPP_PYTHON_BINDINGS=ON"
  ./cbuild.sh --build-type Release --cmake-params "-DCPP_LIBNN=ON -DCPP_PYTHON_BINDINGS=ON"
  cd ../..
  ```

  If any error or missing dependencies please look at the instructions [here](https://github.com/azimonti/ma-libs)


6. Run the program
  ```
  python main.py --mode=play     # default - user plays the game
  python main.py --mode=nn_train # optional --continue for training the neural network
  python main.py --mode=nn_play  # nn is playing the game using the last save neural network
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or want to get in touch regarding the project, please open an issue or contact the repository maintainers directly through GitHub.
