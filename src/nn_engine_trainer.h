#ifndef _NN_ENGINE_TRAINER_H_2295DCD7E7594F7F993A3590FB5D67D2_
#define _NN_ENGINE_TRAINER_H_2295DCD7E7594F7F993A3590FB5D67D2_

/***********************/
/* nn_engine_trainer.h */
/*    Version 1.0      */
/*     2023/05/08      */
/***********************/

#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "ann_mlp_ga_v1.h"
#include "config_cpp.h"

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
        NNEngineTrainer(bool verbose_override = false);
        ~NNEngineTrainer() = default;

        bool init();
        bool load(const std::string& step = "last");
        bool save();
        void train();

        // Optional: A method to get an action from the best network, similar to Python's get_action
        // int get_action(const std::vector<T>& current_state);

      private:
        // Helper methods (definitions in .cpp)
        std::string format_time(long long total_seconds);
        void balance_layouts(std::vector<LayoutInfo>& layouts, int num_layouts_to_balance, double target_ltr_ratio);

        // The core training logic for a single generation
        void train_generation(
            const std::vector<LayoutInfo>& layouts, size_t population_size,
            std::vector<double>& fitness_scores,    // Output: per member total fitness (double for precision)
            std::vector<double>& all_member_steps); // Output: per member total steps (double for precision)

        // Member variables
        std::unique_ptr<nn::ANN_MLP_GA<T>> net_;
        int current_generation_; // Tracks the number of generations completed
        std::chrono::steady_clock::time_point overall_start_time_;
        std::mt19937 random_generator_; // For layout generation or other internal random needs

        // Cached config values from Config:: namespace
        bool verbose_flag_;
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
        bool mixed_population_config_;
        std::vector<size_t> nn_size_config_;

        // Cached game/screen config values
        double cfg_width_;
        double cfg_height_;
        double game_cfg_spad_width_;
        double game_cfg_lpad_width_;
        int game_cfg_max_steps_;
    };

} // namespace Training

#endif
