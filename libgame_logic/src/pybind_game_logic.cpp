/*************************/
/* pybind_game_logic.cpp */
/*     Version 1.0       */
/*      2023/05/05       */
/*************************/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For vector binding
#include "game_logic.h"   // Include the templated class definition

namespace py = pybind11;

// Helper function to bind the templated class
template <typename T> void bind_game_logic_class(py::module& m, const std::string& class_name)
{
    using Class = GameLogicCpp<T>; // Alias for the specific instantiation

    py::class_<Class>(m, class_name.c_str())
        .def(py::init<bool>(), py::arg("no_print_flag") = false) // Constructor binding

        // --- Bind Public Member Variables ---
        .def_readwrite("x", &Class::x)
        .def_readwrite("y", &Class::y)
        .def_readwrite("vx", &Class::vx)
        .def_readwrite("vy", &Class::vy)
        .def_readwrite("fuel", &Class::fuel)
        .def_readwrite("landed", &Class::landed)
        .def_readwrite("crashed", &Class::crashed)
        .def_readwrite("landed_successfully", &Class::landed_successfully)
        .def_readwrite("landing_pad_center_x", &Class::landing_pad_center_x)
        .def_readwrite("landing_pad_y", &Class::landing_pad_y)
        .def_readwrite("no_print", &Class::no_print)
        .def_readwrite("last_action", &Class::last_action)

        // --- Bind Public Methods ---
        .def("set_config", &Class::set_config, py::arg("cfg_w"), py::arg("cfg_h"), py::arg("gcfg_pad_y1"),
             py::arg("gcfg_terrain_y_val"), py::arg("gcfg_max_v_x"), py::arg("gcfg_max_v_y"), py::arg("pcfg_gravity"),
             py::arg("pcfg_fric_x"), py::arg("pcfg_fric_y"), py::arg("lcfg_w"), py::arg("lcfg_h"), py::arg("lcfg_fuel"),
             py::arg("gcfg_spad_width"), py::arg("gcfg_lpad_width"), py::arg("gcfg_x0_vec"), py::arg("gcfg_v0_vec"),
             py::arg("gcfg_a0_vec"), "Sets the configuration parameters for the game simulation.")

        .def("update_pad_positions", &Class::update_pad_positions, py::arg("spad_x1"), py::arg("lpad_x1_new"),
             "Updates the landing pad position dynamically.")

        // Bind reset methods using static_cast to resolve overloads
        .def("reset", static_cast<void (Class::*)()>(&Class::reset),
             "Resets the game state to the initial configuration.")
        .def("reset", static_cast<void (Class::*)(T, T)>(&Class::reset), py::arg("spad_x1"), py::arg("lpad_x1_new"),
             "Resets the game state with specific pad positions.")

        .def("is_done", &Class::is_done, "Checks if the game episode has finished (landed or crashed).")

        // Takes a NumPy array (output buffer) as input, returns void
        .def("get_state",
             [](const Class& self, py::array_t<T> state_output) {
        py::buffer_info buf_info = state_output.request();
        if (buf_info.ndim != 1) { throw std::runtime_error("get_state: Output array must be 1-dimensional."); }
        T* ptr      = static_cast<T*>(buf_info.ptr);
        size_t size = static_cast<size_t>(buf_info.shape[0]);
        // Call the C++ function that writes directly to the buffer
        self.get_state(ptr, size);
    },
             py::arg("state_output"), // The NumPy array to write into
             "Fills the provided NumPy array with the current state vector.")

        // Takes action and a NumPy array (output buffer), returns bool (done flag)
        .def("update",
             [](Class& self, int action, py::array_t<T> state_output) -> bool {
        py::buffer_info buf_info = state_output.request();
        if (buf_info.ndim != 1) { throw std::runtime_error("update: Output state array must be 1-dimensional."); }
        T* ptr      = static_cast<T*>(buf_info.ptr);
        size_t size = static_cast<size_t>(buf_info.shape[0]);
        // Call the C++ function that writes state to buffer and returns done flag
        return self.update(action, ptr, size);
    }, py::arg("action"), py::arg("state_output"), // Action and the NumPy array to write state into
             "Performs one time step, fills the provided NumPy array with the new state, and returns the done flag "
             "(bool).")

        // --- Bind Penalty Calculation Methods ---
        .def("calculate_step_penalty", &Class::calculate_step_penalty, py::arg("action"),
             "Calculates the penalty applied at each step based on state and action.")
        .def("calculate_terminal_penalty", &Class::calculate_terminal_penalty, py::arg("steps_taken"),
             "Calculates the terminal penalty/reward based on the final state and steps taken.");
}

PYBIND11_MODULE(cpp_game_logic, m)
{
    m.doc() = "pybind11 plugin for C++ Lunar Lander Game Logic (float and double versions)";

    // Bind the float version
    bind_game_logic_class<float>(m, "GameLogicCppFloat");

    // Bind the double version
    bind_game_logic_class<double>(m, "GameLogicCppDouble");
}
