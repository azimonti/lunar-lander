/************************/
/* lunar_lander_cpp.cpp */
/*     Version 1.0      */
/*      2023/05/05      */
/************************/

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game_logic.h"
#include "game_utils.h"

namespace py = pybind11;

// Helper function to bind the templated class
template <typename T> void bind_game_logic_class(py::module& m, const std::string& class_name)
{
    using Class = GameLogicCpp<T>; // Alias for the specific instantiation

    py::class_<Class>(m, class_name.c_str())
        .def(py::init<bool, const std::string&>(), py::arg("no_print_flag") = false, py::arg("config_file") = "")

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
        // Binding for the original calculate_terminal_penalty (for Python use)
        .def("calculate_terminal_penalty", static_cast<T (Class::*)(int) const>(&Class::calculate_terminal_penalty),
             py::arg("steps_taken"),
             "Calculates the terminal penalty/reward based on the final state and steps taken (original version).")
        // Binding for the overloaded calculate_terminal_penalty (for L/R bonus tracking, distinct Python name)
        .def("calculate_terminal_penalty_lr_bonus",
             [](const Class& self, int steps_taken, size_t direction) -> py::tuple {
        std::array<T, 2> landing_bonus_lr = {T(0.0), T(0.0)}; // Initialize array
        // Call the C++ overloaded function
        T penalty                         = self.calculate_terminal_penalty(steps_taken, direction, landing_bonus_lr);
        // Return penalty and the (potentially modified) bonus array as a tuple
        return py::make_tuple(penalty, landing_bonus_lr);
    }, py::arg("steps_taken"), py::arg("direction"),
             "Calculates terminal penalty, populating L/R bonus array. Returns (penalty, [bonus_L, bonus_R]).");
}

PYBIND11_MODULE(lunar_lander_cpp, m)
{
    m.doc() = "pybind11 plugin for C++ Lunar Lander";

    // Bind the float version
    bind_game_logic_class<float>(m, "GameLogicCppFloat");

    // Bind the double version
    bind_game_logic_class<double>(m, "GameLogicCppDouble");

    // Bind the utility function
    m.def("generate_random_pad_positions", &generate_random_pad_positions, py::arg("screen_width"),
          py::arg("spad_width"), py::arg("lpad_width"),
          py::arg("seed")                = py::none(), // py::none() maps to std::nullopt for std::optional<int>
          py::arg("force_left_to_right") = false, py::arg("force_right_to_left") = false,
          "Generates random, non-overlapping positions for start and landing pads.");
}
