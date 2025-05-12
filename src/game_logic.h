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

    // NN Training Penalty Config values
    T nn_train_sp_dist_factor;
    T nn_train_sp_action_factor;
    T nn_train_tp_steps_factor;
    T nn_train_tp_dist_factor;
    T nn_train_tp_landed_bonus;
    T nn_train_tp_landed_lr_bonus;
    T nn_train_tp_fuel_bonus_factor;
    T nn_train_tp_crashed_penalty;
    T nn_train_tp_crash_v_mag_factor;
    T nn_train_tp_no_fuel_penalty;

    // --- Private Helper Methods ---
    void apply_action(int action);
    void update_physics();
    void check_landing_crash();

  public:
    // --- Constructor ---
    // If config_file is empty, assumes config is already loaded.
    // Otherwise, loads the specified config file.
    GameLogicCpp(bool no_print_flag = false, const std::string& config_file = "");

    // --- Public Methods ---
    void reset(); // Consider passing config struct here if dynamic
    // Overload reset to accept specific pad positions and initial state if needed for training
    void reset(T spad_x1_new, T lpad_x1_new);

    // The main update function (Returning vector might be less efficient for templates, consider pointer version)
    std::pair<std::vector<T>, bool> update(int action);
    // New update function: writes state to provided buffer, returns done flag
    bool update(int action, T* pStateOutput, size_t stateOutputSize);

    bool is_done() const; // Make const as it doesn't change state

    std::vector<T> get_state() const;
    void get_state(T* pOutputs, size_t outputsSize) const;

    // --- Penalty Calculation Methods ---
    T calculate_step_penalty(int action) const;
    T calculate_terminal_penalty(int steps_taken) const;
    T calculate_terminal_penalty(int steps_taken, size_t direction, std::array<T, 2>& landing_bonus_lr) const;
    int calculate_combined_landing_nb(double total_landing_bonus_lr) const;

  private:
    // Helper to recalculate derived values after config/pad changes
    void recalculate_derived_values();

    // Helper to load configuration vectors with the class's type T
    static std::vector<T> getConfigVectorAsType(const std::string& key);
    // initialize_from_config removed
};

#endif
