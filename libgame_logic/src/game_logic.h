#ifndef _GAME_LOGIC_H_87DBA366D1A949809C99341B7769AAC9_
#define _GAME_LOGIC_H_87DBA366D1A949809C99341B7769AAC9_

/******************/
/*  game_logic.h  */
/*  Version 1.0   */
/*   2023/05/05   */
/******************/

#include <cmath>
#include <vector>

// Forward declaration if config struct/class is used directly
// Or include config header if available and suitable
// For now, assume config values are passed during init or reset

class GameLogicCpp
{
  public:
    double x, y, vx, vy, fuel;
    bool landed, crashed, landed_successfully;
    double landing_pad_center_x, landing_pad_y;
    bool no_print;   // For optional printing
    int last_action; // Added to store the last action taken

    // --- Configuration parameters (to be set in constructor/reset) ---
  private:
    // Physics/State
    double ax, ay; // Internal acceleration state
    int time_step;

    // Cached Config values (set during reset/config)
    double initial_x, initial_y, initial_vx, initial_vy, initial_ax, initial_ay; // Base initial state
    double pcfg_g;
    double lcfg_max_fuel;
    double lpad_x1, lpad_width, pad_y1; // Landing pad
    double spad_width;                  // Starting pad width (needed for reset calculation)
    double max_safe_vx, max_safe_vy;    // Store absolute values
    double cfg_width, cfg_height;
    double lcfg_width, lcfg_height; // Lander dimensions
    double pcfg_mu_x, pcfg_mu_y;
    double gcfg_terrain_y;
    double ground_level; // Calculated in reset

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
    void reset(double spad_x1, double lpad_x1_new);

    // The main update function
    std::pair<std::vector<double>, bool> update(int action);
    // New update function: writes state to provided buffer, returns done flag
    bool update(int action, double* pStateOutput, size_t stateOutputSize);

    bool is_done() const; // Make const as it doesn't change state

    std::vector<double> get_state() const;
    void get_state(double* pOutputs, size_t outputsSize) const;

    // Method to explicitly set config values if not passed in constructor
    void set_config(double cfg_w, double cfg_h, double gcfg_pad_y1, double gcfg_terrain_y_val, double gcfg_max_v_x,
                    double gcfg_max_v_y, double pcfg_gravity, double pcfg_fric_x, double pcfg_fric_y, double lcfg_w,
                    double lcfg_h, double lcfg_fuel,
                    double gcfg_spad_width, // Starting pad width
                    double gcfg_lpad_width, // Landing pad width
                    const std::vector<double>& gcfg_x0_vec, const std::vector<double>& gcfg_v0_vec,
                    const std::vector<double>& gcfg_a0_vec);

    // Method to update pad positions dynamically (used in training)
    void update_pad_positions(double spad_x1, double lpad_x1_new);

  private:
    // Helper to recalculate derived values after config/pad changes
    void recalculate_derived_values();
};

#endif
