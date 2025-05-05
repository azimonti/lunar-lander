#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for automatic vector/pair conversions
#include <pybind11/numpy.h> // Needed for NumPy array conversions
#include "game_logic.h"   // Include the C++ class definition

namespace py = pybind11;

// --- Helper function for get_state binding ---
py::array_t<double> get_state_wrapper(const GameLogicCpp &self) {
    const std::vector<double>& vec = self.get_state(); // Get the vector
    // Convert vector to numpy array directly here
    py::array_t<double> state_np(vec.size());
    double* ptr = static_cast<double*>(state_np.request().ptr);
    std::memcpy(ptr, vec.data(), vec.size() * sizeof(double));
    return state_np; // Return the numpy array
}

// --- Helper function for update binding ---
py::tuple update_wrapper(GameLogicCpp &self, int action) {
    std::pair<std::vector<double>, bool> cpp_result = self.update(action);
    const std::vector<double>& vec = cpp_result.first; // Reference to the vector part
    // Convert vector to numpy array directly here
    py::array_t<double> state_np(vec.size());
    double* ptr = static_cast<double*>(state_np.request().ptr);
    std::memcpy(ptr, vec.data(), vec.size() * sizeof(double));
    // Return the tuple
    return py::make_tuple(state_np, cpp_result.second);
}


PYBIND11_MODULE(cpp_game_logic, m) { // Module name must match CMake target
    m.doc() = "pybind11 plugin for C++ Lunar Lander Game Logic"; // Optional module docstring

    py::class_<GameLogicCpp>(m, "GameLogicCpp")
        .def(py::init<bool>(), py::arg("no_print_flag") = false) // Constructor binding

        // --- Bind Public Member Variables ---
        // Make them accessible as properties from Python
        .def_readwrite("x", &GameLogicCpp::x)
        .def_readwrite("y", &GameLogicCpp::y)
        .def_readwrite("vx", &GameLogicCpp::vx)
        .def_readwrite("vy", &GameLogicCpp::vy)
        .def_readwrite("fuel", &GameLogicCpp::fuel)
        .def_readwrite("landed", &GameLogicCpp::landed)
        .def_readwrite("crashed", &GameLogicCpp::crashed)
        .def_readwrite("landed_successfully", &GameLogicCpp::landed_successfully)
        .def_readwrite("landing_pad_center_x", &GameLogicCpp::landing_pad_center_x)
        .def_readwrite("landing_pad_y", &GameLogicCpp::landing_pad_y)
        .def_readwrite("no_print", &GameLogicCpp::no_print)
        .def_readwrite("last_action", &GameLogicCpp::last_action)
        // Note: Internal C++ variables (ax, ay, time_step, etc.) are not exposed

        // --- Bind Public Methods ---
        .def("set_config", &GameLogicCpp::set_config,
             py::arg("cfg_w"), py::arg("cfg_h"),
             py::arg("gcfg_pad_y1"), py::arg("gcfg_terrain_y_val"),
             py::arg("gcfg_max_v_x"), py::arg("gcfg_max_v_y"),
             py::arg("pcfg_gravity"), py::arg("pcfg_fric_x"), py::arg("pcfg_fric_y"),
             py::arg("lcfg_w"), py::arg("lcfg_h"), py::arg("lcfg_fuel"),
             py::arg("gcfg_spad_width"), py::arg("gcfg_lpad_width"),
             py::arg("gcfg_x0_vec"), py::arg("gcfg_v0_vec"), py::arg("gcfg_a0_vec"),
             "Sets the configuration parameters for the game simulation.")

        .def("update_pad_positions", &GameLogicCpp::update_pad_positions,
             py::arg("spad_x1"), py::arg("lpad_x1_new"),
             "Updates the landing pad position dynamically.")

        // Bind reset methods
        .def("reset", static_cast<void (GameLogicCpp::*)()>(&GameLogicCpp::reset),
             "Resets the game state to the initial configuration.")
        .def("reset", static_cast<void (GameLogicCpp::*)(double, double)>(&GameLogicCpp::reset),
             py::arg("spad_x1"), py::arg("lpad_x1_new"),
             "Resets the game state with specific pad positions.")

        .def("is_done", &GameLogicCpp::is_done,
             "Checks if the game episode has finished (landed or crashed).")

        // Bind get_state using the lambda-free wrapper function
        .def("get_state", &get_state_wrapper,
             "Returns the current state vector as a NumPy array.")

        // Bind update using the lambda-free wrapper function
        .def("update", &update_wrapper, py::arg("action"),
              "Performs one time step of the game simulation and returns (state_numpy_array, done_bool).");
}
