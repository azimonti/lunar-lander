'''
/******************/
/* mod_nn_play.py */
/*  Version 1.0   */
/*   2025/05/08   */
/******************/
'''
import numpy as np
import os
import sys
import time

from mod_config import nn_config

cwd = os.getcwd()
build_dir = os.path.join(cwd, "./externals/ma-libs/build/Release")
sys.path.append(os.path.realpath(build_dir))

try:
    import cpp_nn_py as cpp_nn_py
except ModuleNotFoundError as e:
    print(f"Error: {e}")


class NeuralNetwork():
    def __init__(self):
        # Always loading the double version in play mode
        self._net = cpp_nn_py.ANN_MLP_GA_double()
        self._net.SetName(nn_config.name)
        filename = f"{nn_config.name}_last.txt"

        full_path = os.path.join(nn_config.save_path_nn, filename)
        self._net.Deserialize(full_path)
        # Get current epoch/generation from loaded net
        self._nGen = self._net.GetEpochs()
        self._start_time = time.time()  # Reset start time on load
        self._nnsize = self._net.GetNetworkSize()

    def get_action(self, current_state: np.ndarray) -> int:
        """Gets the action from the NN based on the current state."""
        if self._net is None:
            print("Error: Network not loaded or initialized.")
            # Return a default action (e.g., Noop) or raise an error
            return 0

        if self._nnsize is None:
            print("Error: Network size not determined (load network first).")
            return 0

        inputs = np.array(current_state, dtype=np.float64)

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
