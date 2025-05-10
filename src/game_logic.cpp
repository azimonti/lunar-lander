/********************/
/*  game_logic.cpp  */
/*   Version 1.1    */
/*    2023/05/10    */
/********************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "game_logic.h"

// --- Static Helper Function Specializations ---
// Specialization for GameLogicCpp<double>
template <> std::vector<double> GameLogicCpp<double>::getConfigVectorAsType(const std::string& key)
{
    std::vector<double> vec = Config::getVectorDouble(key);
    return vec;
}

// Specialization for GameLogicCpp<float>
template <> std::vector<float> GameLogicCpp<float>::getConfigVectorAsType(const std::string& key)
{
    return Config::getVectorFloat(key);
}

// --- Constructor ---
template <typename T>
GameLogicCpp<T>::GameLogicCpp(bool no_print_flag_param, const std::string& config_file)
    : // Initialize non-config state variables directly
      x(T(0.0)), y(T(0.0)), vx(T(0.0)), vy(T(0.0)), fuel(T(0.0)), landed(false), crashed(false),
      landed_successfully(false), landing_pad_center_x(T(0.0)), no_print(no_print_flag_param), last_action(0),
      ax(T(0.0)), ay(T(0.0)), time_step(0)
{
    // If config_file is not empty, load it. Otherwise, assume config is pre-loaded.
    if (!config_file.empty())
    {
        if (!Config::loadConfiguration(config_file))
        {
            throw std::runtime_error("FATAL: Could not load configuration from " + config_file + ". Exiting.");
        }
    }
    // Load fixed configuration values from Config system
    pad_y1         = static_cast<T>(Config::getDouble("GameCfg.pad_y1"));
    gcfg_terrain_y = static_cast<T>(Config::getDouble("GameCfg.terrain_y"));

    T max_v_x_raw  = static_cast<T>(Config::getDouble("GameCfg.max_vx"));
    T max_v_y_raw  = static_cast<T>(Config::getDouble("GameCfg.max_vy"));
    max_safe_vx    = std::abs(max_v_x_raw);
    max_safe_vy    = std::abs(max_v_y_raw);
    if (max_safe_vx == T(0.0)) max_safe_vx = T(1e-6);
    if (max_safe_vy == T(0.0)) max_safe_vy = T(1e-6);

    pcfg_g              = static_cast<T>(Config::getDouble("PlanetCfg.g"));
    pcfg_mu_x           = static_cast<T>(Config::getDouble("PlanetCfg.mu_x"));
    pcfg_mu_y           = static_cast<T>(Config::getDouble("PlanetCfg.mu_y"));

    lcfg_width          = static_cast<T>(Config::getDouble("LanderCfg.width"));
    lcfg_height         = static_cast<T>(Config::getDouble("LanderCfg.height"));
    lcfg_max_fuel       = static_cast<T>(Config::getInt("LanderCfg.max_fuel"));

    lpad_x1             = static_cast<T>(Config::getDouble("GameCfg.lpad_x1"));
    lpad_width          = static_cast<T>(Config::getDouble("GameCfg.lpad_width"));

    cfg_width           = static_cast<T>(Config::getDouble("Cfg.width"));
    cfg_height          = static_cast<T>(Config::getDouble("Cfg.height"));

    spad_width          = static_cast<T>(Config::getDouble("GameCfg.spad_width"));

    std::vector<T> x0_T = GameLogicCpp<T>::getConfigVectorAsType("GameCfg.x0");
    std::vector<T> v0_T = GameLogicCpp<T>::getConfigVectorAsType("GameCfg.v0");
    std::vector<T> a0_T = GameLogicCpp<T>::getConfigVectorAsType("GameCfg.a0");

    initial_x           = x0_T[0];
    initial_y           = x0_T[1];
    initial_vx          = v0_T[0];
    initial_vy          = v0_T[1];
    initial_ax          = a0_T[0];
    initial_ay          = a0_T[1];

    recalculate_derived_values(); // Calculate ground level etc.
    reset();                      // Reset game state variables (x, y, fuel, etc.)
}

// initialize_from_config and the other constructor are removed.

template <typename T> void GameLogicCpp<T>::recalculate_derived_values()
{
    // Calculate landing pad center and y-coordinate
    // Ensure lpad_width is valid before calculation
    if (cfg_width > T(0.0) && cfg_height > T(0.0)) // Basic check for world dimensions
    {
        landing_pad_center_x = lpad_x1 + lpad_width / T(2.0);
        landing_pad_y        = pad_y1;

        // Calculate ground level based on current config
        ground_level         = cfg_height - lcfg_height - gcfg_terrain_y;
    }
    else
    {
        // Handle error or set defaults if config is invalid (e.g. cfg_width/height not set yet)
        landing_pad_center_x = T(0.0);
        landing_pad_y        = pad_y1;
        ground_level         = T(0.0);
    }
    // Ensure max fuel is positive
    if (lcfg_max_fuel <= T(0.0)) lcfg_max_fuel = T(1e-6);
}

template <typename T> void GameLogicCpp<T>::reset()
{
    // initial_x/y/vx/vy/ax/ay are now set by constructor
    // pcfg_g and lcfg_max_fuel are set by constructor
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
    // spad_width and lcfg_width should be set (by constructor)
    lpad_x1   = lpad_x1_new; // Update landing pad's X1 position

    // Update lander initial x position based on the new start pad
    initial_x = spad_x1_new + (spad_width / T(2.0)) - (lcfg_width / T(2.0));
    // initial_y, initial_vx, initial_vy, initial_ax, initial_ay remain as set by constructor

    recalculate_derived_values(); // Recalculate based on new lpad_x1
    reset();                      // Call the standard reset logic to reset most other variables (fuel, flags etc.)
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
    case 1: // Thrust Up
        vy -= T(0.3);
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
    default: break;
    }
    fuel = std::max(T(0.0), fuel);
}

template <typename T> void GameLogicCpp<T>::update_physics()
{
    // ay already includes gravity from reset()
    vy += ay;

    vx *= (T(1.0) - pcfg_mu_x); // pcfg_mu_x from constructor
    vy *= (T(1.0) - pcfg_mu_y); // pcfg_mu_y from constructor

    x += vx;
    y += vy;
}

template <typename T> void GameLogicCpp<T>::check_landing_crash()
{
    if (time_step == 0) { return; }

    // ground_level, lpad_x1, lpad_width, lcfg_width, max_safe_vx, max_safe_vy are all set
    if (y >= ground_level)
    {
        y                   = ground_level;
        landed              = true;

        bool on_landing_pad = (x >= lpad_x1 && (x + lcfg_width) <= (lpad_x1 + lpad_width));
        bool safe_speed     = (std::abs(vx) < max_safe_vx && std::abs(vy) < max_safe_vy);

        if (on_landing_pad && safe_speed)
        {
            landed_successfully = true;
            vx                  = T(0.0);
            vy                  = T(0.0);
            if (!no_print) { std::cout << "GameLogicCpp: Successful landing!" << std::endl; }
        }
        else
        {
            crashed             = true;
            landed_successfully = false;
            vx                  = T(0.0);
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
    if (is_done()) { return {get_state(), true}; }

    apply_action(action);
    update_physics();
    check_landing_crash();

    time_step++;
    return {get_state(), is_done()};
}

template <typename T> bool GameLogicCpp<T>::is_done() const
{
    return landed || crashed;
}

template <typename T> std::vector<T> GameLogicCpp<T>::get_state() const
{
    // cfg_width, cfg_height, max_safe_vx, max_safe_vy, lcfg_max_fuel are set
    // landing_pad_center_x, landing_pad_y are set by recalculate_derived_values
    T dist_target_x      = x - landing_pad_center_x;
    T dist_target_y      = y - landing_pad_y;

    std::vector<T> state = {vx / max_safe_vx, vy / max_safe_vy, dist_target_x / cfg_width, dist_target_y / cfg_height,
                            fuel / lcfg_max_fuel};

    state[0]             = std::clamp(state[0], T(-2.0), T(2.0));
    state[1]             = std::clamp(state[1], T(-2.0), T(2.0));
    state[2]             = std::clamp(state[2], T(-5.0), T(5.0));
    state[3]             = std::clamp(state[3], T(-5.0), T(5.0));
    return state;
}

template <typename T> void GameLogicCpp<T>::get_state(T* pOutputs, size_t outputsSize) const
{
    assert(outputsSize >= 5 && "GameLogicCpp<T>::get_state: Output buffer size must be at least 5.");
    (void)outputsSize;

    T dist_target_x = x - landing_pad_center_x;
    T dist_target_y = y - landing_pad_y;

    pOutputs[0]     = vx / max_safe_vx;
    pOutputs[1]     = vy / max_safe_vy;
    pOutputs[2]     = dist_target_x / cfg_width;
    pOutputs[3]     = dist_target_y / cfg_height;
    pOutputs[4]     = fuel / lcfg_max_fuel;

    pOutputs[0]     = std::clamp(pOutputs[0], T(-2.0), T(2.0));
    pOutputs[1]     = std::clamp(pOutputs[1], T(-2.0), T(2.0));
    pOutputs[2]     = std::clamp(pOutputs[2], T(-5.0), T(5.0));
    pOutputs[3]     = std::clamp(pOutputs[3], T(-5.0), T(5.0));
}

template <typename T> bool GameLogicCpp<T>::update(int action, T* pStateOutput, size_t stateOutputSize)
{
    if (is_done())
    {
        get_state(pStateOutput, stateOutputSize);
        return true;
    }

    apply_action(action);
    update_physics();
    check_landing_crash();

    time_step++;
    get_state(pStateOutput, stateOutputSize);
    return is_done();
}

template <typename T> T GameLogicCpp<T>::calculate_step_penalty(int action) const
{
    T step_penalty = T(0.0);
    T dist_x       = std::abs(x - landing_pad_center_x);
    T dist_y       = std::abs(y - landing_pad_y);
    step_penalty += (dist_x + dist_y) * T(0.001);
    if (action > 0) { step_penalty += T(0.01); }
    return step_penalty;
}

template <typename T> T GameLogicCpp<T>::calculate_terminal_penalty(int steps_taken) const
{
    T terminal_penalty = T(0.0);
    terminal_penalty += static_cast<T>(steps_taken) * T(0.1);

    T dist_x = std::abs(x - landing_pad_center_x);
    T dist_y = std::abs(y - landing_pad_y);
    terminal_penalty += (dist_x + dist_y) * T(0.5);

    if (landed_successfully)
    {
        terminal_penalty -= T(1000.0);
        terminal_penalty -= fuel * T(2.0);
    }
    else if (crashed)
    {
        terminal_penalty += T(500.0);
        T final_v_mag = std::sqrt(vx * vx + vy * vy);
        terminal_penalty += final_v_mag * T(10.0);
    }
    else if (fuel <= T(0.0) && !landed) { terminal_penalty += T(250.0); }

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
