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
GameLogicCpp::GameLogicCpp(bool no_print_flag)
    : // Initialize state variables to reasonable defaults
      x(0.0), y(0.0), vx(0.0), vy(0.0), fuel(0.0), landed(false), crashed(false), landed_successfully(false),
      landing_pad_center_x(0.0), landing_pad_y(0.0), no_print(no_print_flag), last_action(0), // Default action is Noop
      // Internal state
      ax(0.0), ay(0.0), time_step(0),
      // Config defaults (will be overwritten by set_config)
      initial_x(0.0), initial_y(0.0), initial_vx(0.0), initial_vy(0.0), initial_ax(0.0), initial_ay(0.0), pcfg_g(0.0),
      lcfg_max_fuel(100.0),                       // Default value
      lpad_x1(0.0), lpad_width(0.0), pad_y1(0.0), // Landing pad
      spad_width(80.0),                           // Default starting pad width
      max_safe_vx(1.0), max_safe_vy(1.0),         // Default safe values > 0
      cfg_width(100.0), cfg_height(100.0),        // Default dimensions
      lcfg_width(10.0), lcfg_height(10.0),        // Default lander dimensions
      pcfg_mu_x(0.0), pcfg_mu_y(0.0), gcfg_terrain_y(0.0), ground_level(0.0)
{
    // reset(); // Reset is called after set_config typically
}

// --- Configuration Setting ---
void GameLogicCpp::set_config(double cfg_w, double cfg_h, double gcfg_pad_y1, double gcfg_terrain_y_val,
                              double gcfg_max_v_x, double gcfg_max_v_y, double pcfg_gravity, double pcfg_fric_x,
                              double pcfg_fric_y, double lcfg_w, double lcfg_h, double lcfg_fuel,
                              double gcfg_spad_width, double gcfg_lpad_width, const std::vector<double>& gcfg_x0_vec,
                              const std::vector<double>& gcfg_v0_vec, const std::vector<double>& gcfg_a0_vec)
{
    if (gcfg_x0_vec.size() != 2 || gcfg_v0_vec.size() != 2 || gcfg_a0_vec.size() != 2)
    {
        throw std::runtime_error("GameLogicCpp::set_config: Initial state vectors must have size 2.");
    }

    cfg_width      = cfg_w;
    cfg_height     = cfg_h;
    pad_y1         = gcfg_pad_y1;
    gcfg_terrain_y = gcfg_terrain_y_val;

    // Store absolute values for max speeds
    max_safe_vx    = std::abs(gcfg_max_v_x);
    max_safe_vy    = std::abs(gcfg_max_v_y);
    // Prevent division by zero in get_state
    if (max_safe_vx == 0.0) max_safe_vx = 1e-6;
    if (max_safe_vy == 0.0) max_safe_vy = 1e-6;

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

void GameLogicCpp::update_pad_positions(double spad_x1, double lpad_x1_new)
{
    // Assuming spad_x1 is not used based on Python logic, but keeping param
    // Assuming lpad_width is fixed and set during set_config or has a default
    // If lpad_width needs to be passed too, modify the signature
    this->lpad_x1 = lpad_x1_new;
    (void)spad_x1;
    // Need lpad_width to be set before calling this
    recalculate_derived_values();
}

void GameLogicCpp::recalculate_derived_values()
{
    // Calculate landing pad center and y-coordinate
    // Ensure lpad_width is valid before calculation
    if (cfg_width > 0 && cfg_height > 0)
    { // Basic check
        landing_pad_center_x = lpad_x1 + lpad_width / 2.0;
        landing_pad_y        = pad_y1; // Pad Y seems fixed relative to terrain/config

        // Calculate ground level based on current config
        ground_level         = cfg_height - lcfg_height - gcfg_terrain_y;
    }
    else
    {
        // Handle error or set defaults if config is invalid
        landing_pad_center_x = 0.0;
        landing_pad_y        = 0.0;
        ground_level         = 0.0;
        if (!no_print)
        {
            std::cerr << "Warning: Invalid config dimensions in recalculate_derived_values." << std::endl;
        }
    }
    // Ensure max fuel is positive
    if (lcfg_max_fuel <= 0) lcfg_max_fuel = 1e-6;
}

void GameLogicCpp::reset()
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
void GameLogicCpp::reset(double spad_x1, double lpad_x1_new)
{
    update_pad_positions(spad_x1, lpad_x1_new); // Update landing pad info first

    // Call the standard reset logic to reset most variables (vy, fuel, flags etc.)
    reset();

    // Calculate the lander's starting x to be centered on the start pad
    // This mirrors the logic in Python's reset_pad_positions
    x = spad_x1 + (spad_width / 2.0) - (lcfg_width / 2.0);
    // Keep the y position from the standard reset (initial_y)
}

void GameLogicCpp::apply_action(int action)
{
    last_action = action;

    if (fuel <= 0.0)
    {
        return; // No thrust if out of fuel
    }

    switch (action)
    {
    case 1:        // Thrust Up
        vy -= 0.3; // Instantaneous change, adjust as needed (Matches Python)
        fuel -= 1.0;
        break;
    case 2: // Thrust Left
        vx -= 0.05;
        fuel -= 0.5;
        break;
    case 3: // Thrust Right
        vx += 0.05;
        fuel -= 0.5;
        break;
    case 0: // Noop
    default:
        // Do nothing
        break;
    }

    // Clamp fuel to zero
    fuel = std::max(0.0, fuel);
}

void GameLogicCpp::update_physics()
{
    // Update velocity with acceleration (gravity is already in ay)
    // vx += ax; // No horizontal acceleration
    vy += ay;

    // Apply friction/air resistance
    vx *= (1.0 - pcfg_mu_x);
    vy *= (1.0 - pcfg_mu_y);

    // Update position
    x += vx;
    y += vy;

    // Ground collision is handled in check_landing_crash
}

void GameLogicCpp::check_landing_crash()
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
            vx                  = 0.0; // Stop movement on successful landing
            vy                  = 0.0;
            if (!no_print) { std::cout << "GameLogicCpp: Successful landing!" << std::endl; }
        }
        else
        {
            crashed             = true;
            landed_successfully = false;
            vx                  = 0.0; // Stop movement on crash
            vy                  = 0.0;
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

std::pair<std::vector<double>, bool> GameLogicCpp::update(int action)
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

    bool done                 = is_done();
    std::vector<double> state = get_state();

    return {state, done};
}

bool GameLogicCpp::is_done() const
{
    // Episode ends if landed or crashed
    return landed || crashed;
}

std::vector<double> GameLogicCpp::get_state() const
{
    // Calculate distance to target landing pad center
    double dist_target_x      = x - landing_pad_center_x;
    // Target Y is the top of the pad (landing_pad_y)
    // Distance is current Y relative to pad Y. Positive means above.
    double dist_target_y      = y - landing_pad_y;

    // Normalize the state for the NN
    // Ensure denominators are not zero (handled in set_config/recalculate)
    std::vector<double> state = {
        vx / max_safe_vx,           // Normalized Vx
        vy / max_safe_vy,           // Normalized Vy
        dist_target_x / cfg_width,  // Normalized distance X
        dist_target_y / cfg_height, // Normalized distance Y (relative to screen height)
        fuel / lcfg_max_fuel        // Normalized Fuel
    };

    // Apply clipping similar to Python
    state[0] = std::clamp(state[0], -2.0, 2.0); // vx_norm
    state[1] = std::clamp(state[1], -2.0, 2.0); // vy_norm
    state[2] = std::clamp(state[2], -5.0, 5.0); // dist_x_norm
    state[3] = std::clamp(state[3], -5.0, 5.0); // dist_y_norm
    // Fuel is already clamped between 0 and 1 implicitly by normalization

    return state;
}

// New get_state overload: Writes directly to the provided buffer
void GameLogicCpp::get_state(double* pOutputs, size_t outputsSize) const
{

    assert(outputsSize >= 5 && "GameLogicCpp::get_state: Output buffer size must be at least 5.");
    (void)outputsSize;

    // Calculate distance to target landing pad center
    double dist_target_x = x - landing_pad_center_x;
    double dist_target_y = y - landing_pad_y;

    // Calculate normalized values directly into the output buffer
    // Ensure denominators are not zero (handled in set_config/recalculate)
    pOutputs[0]          = vx / max_safe_vx;           // Normalized Vx
    pOutputs[1]          = vy / max_safe_vy;           // Normalized Vy
    pOutputs[2]          = dist_target_x / cfg_width;  // Normalized distance X
    pOutputs[3]          = dist_target_y / cfg_height; // Normalized distance Y
    pOutputs[4]          = fuel / lcfg_max_fuel;       // Normalized Fuel

    // Apply clipping directly to the buffer elements
    pOutputs[0]          = std::clamp(pOutputs[0], -2.0, 2.0); // vx_norm
    pOutputs[1]          = std::clamp(pOutputs[1], -2.0, 2.0); // vy_norm
    pOutputs[2]          = std::clamp(pOutputs[2], -5.0, 5.0); // dist_x_norm
    pOutputs[3]          = std::clamp(pOutputs[3], -5.0, 5.0); // dist_y_norm
}

// New update overload: Writes state to buffer, returns done flag
bool GameLogicCpp::update(int action, double* pStateOutput, size_t stateOutputSize)
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
