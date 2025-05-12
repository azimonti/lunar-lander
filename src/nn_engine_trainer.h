#ifndef _NN_ENGINE_TRAINER_H_2295DCD7E7594F7F993A3590FB5D67D2_
#define _NN_ENGINE_TRAINER_H_2295DCD7E7594F7F993A3590FB5D67D2_

/***********************/
/* nn_engine_trainer.h */
/*    Version 1.0      */
/*     2023/05/08      */
/***********************/

#include <array> // Added for std::array
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "ann_mlp_ga_v1.h"
#include "game_logic.h" // Added for GameLogicCpp<T>

namespace Training
{

    // Structure to hold layout information (used by public train method indirectly)
    struct LayoutInfo
    {
        int spad_x1;
        int lpad_x1;
        double spad_center;
        double lpad_center;
    };

    template <typename T> class NNEngineTrainer
    {
      public:
        NNEngineTrainer();
        ~NNEngineTrainer() = default;

        bool init();
        bool load(const std::string& step = "last");
        bool save();
        void train();

      private:
        // Helper methods (definitions in .cpp)
        std::string format_time(long long total_seconds);
        // Updated signature to include layout directions
        void balance_layouts(std::vector<LayoutInfo>& layouts, std::vector<size_t>& layout_directions,
                             int num_layouts_to_balance, double target_ltr_ratio);

        // The core training logic for a single generation
        void train_generation(
            const std::vector<LayoutInfo>& layouts,
            const std::vector<size_t>& layout_directions, // Added: Directions for each layout
            size_t population_size,
            std::vector<double>& fitness_scores,   // Output: per member total fitness (double for precision)
            std::vector<double>& last_gen_lr,      // Output: per member left right convergence (double for precision)
            std::vector<double>& all_member_steps, // Output: per member total steps (double for precision)
            std::vector<GameLogicCpp<T>>& game_sims); // Added: vector of game simulators

        // Member variables
        std::unique_ptr<nn::ANN_MLP_GA<T>> net_;
        int current_generation_; // Tracks the number of generations completed
        std::chrono::steady_clock::time_point overall_start_time_;
        std::mt19937 random_generator_; // For layout generation or other internal random needs

        std::string nn_name_;
        std::string save_path_nn_;
        bool save_nn_flag_;
        bool overwrite_save_;
        int save_interval_;
        int epochs_; // Total epochs to train for in one call to train()
        int layout_nb_;
        double left_right_ratio_;

        int nn_seed_;
        int population_size_config_;
        int top_individuals_config_;
        int activation_id_config_;
        bool elitism_config_;
        bool multithread_;
        std::vector<size_t> nn_size_config_;

        // Cached game/screen config values
        double cfg_width_;
        double cfg_height_;
        double game_cfg_spad_width_;
        double game_cfg_lpad_width_;
        int game_cfg_max_steps_;

        // For layout reset on stagnation
        int reset_period_config_;
        int generations_stagnated_;
        double last_best_fitness_overall_;

        // Fitness function parameters
        double max_expected_penalty_no_layouts_;

        // GA parameters
        double random_injection_ratio_;
    };

} // namespace Training

#endif
