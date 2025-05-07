#ifndef _GAME_LOGIC_H_87DBA366D1A949809C99341B7769AAC9_
#define _GAME_LOGIC_H_87DBA366D1A949809C99341B7769AAC9_

/******************/
/*  game_logic.h  */
/*  Version 1.0   */
/*   2023/05/05   */
/******************/

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

template <typename T> class GameLogicCpp
{
  public:
    T x, y, vx, vy, fuel;
    bool landed, crashed, landed_successfully;
    T landing_pad_center_x, landing_pad_y;
    bool no_print;   // For optional printing
    int last_action; // Added to store the last action taken

    // --- Configuration parameters (to be set in constructor/reset) ---
  private:
    // Physics/State
    T ax, ay; // Internal acceleration state
    int time_step;

    // Cached Config values (set during reset/config)
    T initial_x, initial_y, initial_vx, initial_vy, initial_ax, initial_ay; // Base initial state
    T pcfg_g;
    T lcfg_max_fuel;
    T lpad_x1, lpad_width, pad_y1; // Landing pad
    T spad_width;                  // Starting pad width (needed for reset calculation)
    T max_safe_vx, max_safe_vy;    // Store absolute values
    T cfg_width, cfg_height;
    T lcfg_width, lcfg_height; // Lander dimensions
    T pcfg_mu_x, pcfg_mu_y;
    T gcfg_terrain_y;
    T ground_level; // Calculated in reset

    // --- Private Helper Methods ---
    void apply_action(int action);
    void update_physics();
    void check_landing_crash();

  public:
    // --- Constructor ---
    // Pass necessary config values directly or via a struct/class
    GameLogicCpp(bool no_print_flag = false);

    // --- Public Methods ---
    void reset(); // Consider passing config struct here if dynamic
    // Overload reset to accept specific pad positions if needed for training
    void reset(T spad_x1_new, T lpad_x1_new);

    // The main update function (Returning vector might be less efficient for templates, consider pointer version)
    std::pair<std::vector<T>, bool> update(int action);
    // New update function: writes state to provided buffer, returns done flag
    bool update(int action, T* pStateOutput, size_t stateOutputSize);

    bool is_done() const; // Make const as it doesn't change state

    std::vector<T> get_state() const;
    void get_state(T* pOutputs, size_t outputsSize) const;

    // Method to explicitly set config values if not passed in constructor
    void set_config(T cfg_w, T cfg_h, T gcfg_pad_y1, T gcfg_terrain_y_val, T gcfg_max_v_x, T gcfg_max_v_y,
                    T pcfg_gravity, T pcfg_fric_x, T pcfg_fric_y, T lcfg_w, T lcfg_h, T lcfg_fuel,
                    T gcfg_spad_width, // Starting pad width
                    T gcfg_lpad_width, // Landing pad width
                    const std::vector<T>& gcfg_x0_vec, const std::vector<T>& gcfg_v0_vec,
                    const std::vector<T>& gcfg_a0_vec);

    // --- Penalty Calculation Methods ---
    T calculate_step_penalty(int action) const;
    T calculate_terminal_penalty(int steps_taken) const;

  private:
    // Helper to recalculate derived values after config/pad changes
    void recalculate_derived_values();
};

#endif
