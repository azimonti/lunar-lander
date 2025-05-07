/********************/
/*  game_logic.cpp  */
/*   Version 1.0    */
/*    2023/05/05    */
/********************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "game_logic.h"

// --- Constructor ---
template <typename T>
GameLogicCpp<T>::GameLogicCpp(bool no_print_flag)
    : // Initialize state variables to reasonable defaults
      x(T(0.0)), y(T(0.0)), vx(T(0.0)), vy(T(0.0)), fuel(T(0.0)), landed(false), crashed(false),
      landed_successfully(false), landing_pad_center_x(T(0.0)), landing_pad_y(T(0.0)), no_print(no_print_flag),
      last_action(0), // Default action is Noop
      // Internal state
      ax(T(0.0)), ay(T(0.0)), time_step(0),
      // Config defaults (will be overwritten by set_config)
      initial_x(T(0.0)), initial_y(T(0.0)), initial_vx(T(0.0)), initial_vy(T(0.0)), initial_ax(T(0.0)),
      initial_ay(T(0.0)), pcfg_g(T(0.0)), lcfg_max_fuel(T(100.0)), // Default value
      lpad_x1(T(0.0)), lpad_width(T(0.0)), pad_y1(T(0.0)),         // Landing pad
      spad_width(T(80.0)),                                         // Default starting pad width
      max_safe_vx(T(1.0)), max_safe_vy(T(1.0)),                    // Default safe values > 0
      cfg_width(T(100.0)), cfg_height(T(100.0)),                   // Default dimensions
      lcfg_width(T(10.0)), lcfg_height(T(10.0)),                   // Default lander dimensions
      pcfg_mu_x(T(0.0)), pcfg_mu_y(T(0.0)), gcfg_terrain_y(T(0.0)), ground_level(T(0.0))
{
}

// --- Configuration Setting ---
template <typename T>
void GameLogicCpp<T>::set_config(T cfg_w, T cfg_h, T gcfg_pad_y1, T gcfg_terrain_y_val, T gcfg_max_v_x, T gcfg_max_v_y,
                                 T pcfg_gravity, T pcfg_fric_x, T pcfg_fric_y, T lcfg_w, T lcfg_h, T lcfg_fuel,
                                 T gcfg_spad_width, T gcfg_lpad_width, const std::vector<T>& gcfg_x0_vec,
                                 const std::vector<T>& gcfg_v0_vec, const std::vector<T>& gcfg_a0_vec)
{
    if (gcfg_x0_vec.size() != 2 || gcfg_v0_vec.size() != 2 || gcfg_a0_vec.size() != 2)
    {
        throw std::runtime_error("GameLogicCpp<T>::set_config: Initial state vectors must have size 2.");
    }

    cfg_width      = cfg_w;
    cfg_height     = cfg_h;
    pad_y1         = gcfg_pad_y1;
    gcfg_terrain_y = gcfg_terrain_y_val;

    // Store absolute values for max speeds
    max_safe_vx    = std::abs(gcfg_max_v_x);
    max_safe_vy    = std::abs(gcfg_max_v_y);
    // Prevent division by zero in get_state
    if (max_safe_vx == T(0.0)) max_safe_vx = T(1e-6);
    if (max_safe_vy == T(0.0)) max_safe_vy = T(1e-6);

    pcfg_g        = pcfg_gravity;
    pcfg_mu_x     = pcfg_fric_x;
    pcfg_mu_y     = pcfg_fric_y;

    lcfg_width    = lcfg_w;
    lcfg_height   = lcfg_h;
    lcfg_max_fuel = lcfg_fuel;
    spad_width    = gcfg_spad_width;
    lpad_width    = gcfg_lpad_width;

    initial_x     = gcfg_x0_vec[0];
    initial_y     = gcfg_x0_vec[1];
    initial_vx    = gcfg_v0_vec[0];
    initial_vy    = gcfg_v0_vec[1];
    initial_ax    = gcfg_a0_vec[0];
    initial_ay    = gcfg_a0_vec[1];

    // Initial pad position might need to be set here if not always dynamic

    recalculate_derived_values(); // Calculate ground level etc.
}

template <typename T> void GameLogicCpp<T>::recalculate_derived_values()
{
    // Calculate landing pad center and y-coordinate
    // Ensure lpad_width is valid before calculation
    if (cfg_width > T(0.0) && cfg_height > T(0.0))
    { // Basic check
        landing_pad_center_x = lpad_x1 + lpad_width / T(2.0);
        landing_pad_y        = pad_y1; // Pad Y seems fixed relative to terrain/config

        // Calculate ground level based on current config
        ground_level         = cfg_height - lcfg_height - gcfg_terrain_y;
    }
    else
    {
        // Handle error or set defaults if config is invalid
        landing_pad_center_x = T(0.0);
        landing_pad_y        = T(0.0);
        ground_level         = T(0.0);
        if (!no_print)
        {
            std::cerr << "Warning: Invalid config dimensions in recalculate_derived_values." << std::endl;
        }
    }
    // Ensure max fuel is positive
    if (lcfg_max_fuel <= T(0.0)) lcfg_max_fuel = T(1e-6);
}

template <typename T> void GameLogicCpp<T>::reset()
{
    x                   = initial_x;
    y                   = initial_y;
    vx                  = initial_vx;
    vy                  = initial_vy;
    ax                  = initial_ax;
    ay                  = initial_ay + pcfg_g; // Apply gravity to initial vertical acceleration
    fuel                = lcfg_max_fuel;
    landed              = false;
    crashed             = false;
    landed_successfully = false;
    time_step           = 0;
    last_action         = 0; // Reset last action
}

// Overload reset to accept specific pad positions and calculate initial x
template <typename T> void GameLogicCpp<T>::reset(T spad_x1_new, T lpad_x1_new)
{
    lpad_x1   = lpad_x1_new;
    // Need lpad_width to be set before calling this
    //  Update lander initial x position based on the new start pad
    initial_x = spad_x1_new + (spad_width / T(2.0)) - (lcfg_width / T(2.0));
    // Keep the y position from the standard reset (initial_y)
    // Call the standard reset logic to reset most variables (vy, fuel, flags etc.)
    recalculate_derived_values();
    reset();
}

template <typename T> void GameLogicCpp<T>::apply_action(int action)
{
    last_action = action;

    if (fuel <= T(0.0))
    {
        return; // No thrust if out of fuel
    }

    switch (action)
    {
    case 1:           // Thrust Up
        vy -= T(0.3); // Instantaneous change, adjust as needed (Matches Python)
        fuel -= T(1.0);
        break;
    case 2: // Thrust Left
        vx -= T(0.05);
        fuel -= T(0.5);
        break;
    case 3: // Thrust Right
        vx += T(0.05);
        fuel -= T(0.5);
        break;
    case 0: // Noop
    default:
        // Do nothing
        break;
    }

    // Clamp fuel to zero
    fuel = std::max(T(0.0), fuel);
}

template <typename T> void GameLogicCpp<T>::update_physics()
{
    // Update velocity with acceleration (gravity is already in ay)
    // vx += ax; // No horizontal acceleration
    vy += ay;

    // Apply friction/air resistance
    vx *= (T(1.0) - pcfg_mu_x);
    vy *= (T(1.0) - pcfg_mu_y);

    // Update position
    x += vx;
    y += vy;

    // Ground collision is handled in check_landing_crash
}

template <typename T> void GameLogicCpp<T>::check_landing_crash()
{
    // Only check for landing/crash after the first time step
    if (time_step == 0) { return; }

    // Check if the lander's bottom edge is at or below ground level
    if (y >= ground_level)
    {
        y                   = ground_level; // Ensure it rests exactly on the ground
        landed              = true;

        // Check if horizontally within the landing pad boundaries
        bool on_landing_pad = (x >= lpad_x1 && (x + lcfg_width) <= (lpad_x1 + lpad_width));

        // Check if speeds are within safe limits
        bool safe_speed     = (std::abs(vx) < max_safe_vx && std::abs(vy) < max_safe_vy);

        if (on_landing_pad && safe_speed)
        {
            landed_successfully = true;
            vx                  = T(0.0); // Stop movement on successful landing
            vy                  = T(0.0);
            if (!no_print) { std::cout << "GameLogicCpp: Successful landing!" << std::endl; }
        }
        else
        {
            crashed             = true;
            landed_successfully = false;
            vx                  = T(0.0); // Stop movement on crash
            vy                  = T(0.0);
            if (!no_print)
            {
                if (!on_landing_pad) { std::cout << "GameLogicCpp: Crashed - outside landing area!" << std::endl; }
                if (std::abs(vx) >= max_safe_vx)
                {
                    std::cout << "GameLogicCpp: Crashed - excessive horizontal speed! (|";
                    std::cout << vx << "| >= " << max_safe_vx << ")" << std::endl;
                }
                if (std::abs(vy) >= max_safe_vy)
                {
                    std::cout << "GameLogicCpp: Crashed - excessive vertical speed! (|";
                    std::cout << vy << "| >= " << max_safe_vy << ")" << std::endl;
                }
            }
        }
    }
}

template <typename T> std::pair<std::vector<T>, bool> GameLogicCpp<T>::update(int action)
{
    if (is_done())
    {
        // Return current state and True for done
        return {get_state(), true};
    }

    apply_action(action);
    update_physics();
    check_landing_crash();

    time_step++;

    bool done            = is_done();
    std::vector<T> state = get_state();

    return {state, done};
}

template <typename T> bool GameLogicCpp<T>::is_done() const
{
    // Episode ends if landed or crashed
    return landed || crashed;
}

template <typename T> std::vector<T> GameLogicCpp<T>::get_state() const
{
    // Calculate distance to target landing pad center
    T dist_target_x      = x - landing_pad_center_x;
    // Target Y is the top of the pad (landing_pad_y)
    // Distance is current Y relative to pad Y. Positive means above.
    T dist_target_y      = y - landing_pad_y;

    // Normalize the state for the NN
    // Ensure denominators are not zero (handled in set_config/recalculate)
    std::vector<T> state = {
        vx / max_safe_vx,           // Normalized Vx
        vy / max_safe_vy,           // Normalized Vy
        dist_target_x / cfg_width,  // Normalized distance X
        dist_target_y / cfg_height, // Normalized distance Y (relative to screen height)
        fuel / lcfg_max_fuel        // Normalized Fuel
    };

    // Apply clipping similar to Python
    state[0] = std::clamp(state[0], T(-2.0), T(2.0)); // vx_norm
    state[1] = std::clamp(state[1], T(-2.0), T(2.0)); // vy_norm
    state[2] = std::clamp(state[2], T(-5.0), T(5.0)); // dist_x_norm
    state[3] = std::clamp(state[3], T(-5.0), T(5.0)); // dist_y_norm
    // Fuel is already clamped between 0 and 1 implicitly by normalization

    return state;
}

// New get_state overload: Writes directly to the provided buffer
template <typename T> void GameLogicCpp<T>::get_state(T* pOutputs, size_t outputsSize) const
{

    assert(outputsSize >= 5 && "GameLogicCpp<T>::get_state: Output buffer size must be at least 5.");
    (void)outputsSize;

    // Calculate distance to target landing pad center
    T dist_target_x = x - landing_pad_center_x;
    T dist_target_y = y - landing_pad_y;

    // Calculate normalized values directly into the output buffer
    // Ensure denominators are not zero (handled in set_config/recalculate)
    pOutputs[0]     = vx / max_safe_vx;           // Normalized Vx
    pOutputs[1]     = vy / max_safe_vy;           // Normalized Vy
    pOutputs[2]     = dist_target_x / cfg_width;  // Normalized distance X
    pOutputs[3]     = dist_target_y / cfg_height; // Normalized distance Y
    pOutputs[4]     = fuel / lcfg_max_fuel;       // Normalized Fuel

    // Apply clipping directly to the buffer elements
    pOutputs[0]     = std::clamp(pOutputs[0], T(-2.0), T(2.0)); // vx_norm
    pOutputs[1]     = std::clamp(pOutputs[1], T(-2.0), T(2.0)); // vy_norm
    pOutputs[2]     = std::clamp(pOutputs[2], T(-5.0), T(5.0)); // dist_x_norm
    pOutputs[3]     = std::clamp(pOutputs[3], T(-5.0), T(5.0)); // dist_y_norm
}

// New update overload: Writes state to buffer, returns done flag
template <typename T> bool GameLogicCpp<T>::update(int action, T* pStateOutput, size_t stateOutputSize)
{
    if (is_done())
    {
        // Populate the output buffer with the current state and return true
        get_state(pStateOutput, stateOutputSize); // Use the new get_state overload
        return true;
    }

    apply_action(action);
    update_physics();
    check_landing_crash(); // Handle ground collision and state changes

    time_step++;

    bool done = is_done();
    // Populate the output buffer with the new state
    get_state(pStateOutput, stateOutputSize);

    return done; // Return only the done flag
}

// --- Penalty Calculation Methods ---

template <typename T> T GameLogicCpp<T>::calculate_step_penalty(int action) const
{
    T step_penalty = T(0.0);

    // Penalty for distance from pad center
    T dist_x       = std::abs(x - landing_pad_center_x);
    T dist_y       = std::abs(y - landing_pad_y);
    step_penalty += (dist_x + dist_y) * T(0.001);

    // Penalty for using fuel (if action > 0)
    if (action > 0)
    {
        step_penalty += T(0.01); // Small penalty per thrust action
    }

    return step_penalty;
}

template <typename T> T GameLogicCpp<T>::calculate_terminal_penalty(int steps_taken) const
{
    T terminal_penalty = T(0.0);

    // Base penalty for steps taken (encourages efficiency)
    terminal_penalty += static_cast<T>(steps_taken) * T(0.1);

    // Distance penalty (applied more heavily at the end)
    T dist_x = std::abs(x - landing_pad_center_x);
    T dist_y = std::abs(y - landing_pad_y);
    // Scale distance penalty - higher if further away
    terminal_penalty += (dist_x + dist_y) * T(0.5);

    if (landed_successfully)
    {
        terminal_penalty -= T(1000.0); // Big reward (negative penalty)
        // Bonus for remaining fuel
        terminal_penalty -= fuel * T(2.0);
    }
    else if (crashed)
    {
        terminal_penalty += T(500.0); // Penalty for crashing
        // Increase penalty based on final velocity magnitude
        T final_v_mag = std::sqrt(vx * vx + vy * vy);
        terminal_penalty += final_v_mag * T(10.0);
    }
    else if (fuel <= T(0.0) && !landed)
    {                                 // Check if fuel is exactly 0 or less
        terminal_penalty += T(250.0); // Penalty for running out of fuel in air
    }

    // Add small penalty based on final velocity if not landed successfully
    if (!landed_successfully)
    {
        T final_v_mag = std::sqrt(vx * vx + vy * vy);
        terminal_penalty += final_v_mag * T(1.0);
    }

    return terminal_penalty;
}

// Explicit template instantiation
template class GameLogicCpp<float>;
template class GameLogicCpp<double>;
