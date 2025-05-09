/*************************/
/* nn_engine_trainer.cpp */
/*      Version 1.1      */
/*       2023/05/10      */
/*************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "thread/thread_pool.hpp"
#include "config_loader.h"
#include "game_logic.h"
#include "game_utils.h"
#include "nn_engine_trainer.h"

template <typename T> Training::NNEngineTrainer<T>::NNEngineTrainer() : current_generation_(0)
{
    // Cache configuration values
    nn_name_                = Config::getString("NNConfig.name", "default_nn");
    save_path_nn_           = Config::getString("NNConfig.save_path_nn", "data/nn_models/");
    save_nn_flag_           = Config::getBool("NNConfig.save_nn", true);
    overwrite_save_         = Config::getBool("NNConfig.overwrite", false);
    save_interval_          = Config::getInt("NNConfig.save_interval", 10);
    epochs_                 = Config::getInt("NNConfig.epochs", 100);
    layout_nb_              = Config::getInt("NNConfig.layout_nb", 10);
    left_right_ratio_       = Config::getDouble("NNConfig.left_right_ratio", 0.5);

    nn_seed_                = Config::getInt("NNConfig.seed", 0);
    population_size_config_ = Config::getInt("NNConfig.population_size", 100);
    top_individuals_config_ = Config::getInt("NNConfig.top_individuals", 10);
    activation_id_config_   = Config::getInt("NNConfig.activation_id", 0);
    elitism_config_         = Config::getBool("NNConfig.elitism", true);
    multithread_            = Config::getBool("NNConfig.multithread", true);

    nn_size_config_.push_back(5); // Input layer: 5 state variables
    std::vector<int> hlayers_vec = Config::getVectorInt("NNConfig.hlayers");
    for (int h_size : hlayers_vec) { nn_size_config_.push_back(static_cast<size_t>(h_size)); }
    nn_size_config_.push_back(4); // Output layer: 4 actions

    cfg_width_                 = Config::getDouble("Cfg.width", 800.0);
    cfg_height_                = Config::getDouble("Cfg.height", 600.0);
    game_cfg_spad_width_       = Config::getDouble("GameCfg.spad_width", 100.0);
    game_cfg_lpad_width_       = Config::getDouble("GameCfg.lpad_width", 100.0);
    game_cfg_max_steps_        = Config::getInt("GameCfg.max_steps", 1000);

    // Initialize stagnation tracking variables
    reset_period_config_       = Config::getInt("NNConfig.reset_period", 100000); // Default to a high value
    generations_stagnated_     = 0;
    last_best_fitness_overall_ = std::numeric_limits<double>::max();

    random_generator_.seed(static_cast<unsigned int>(nn_seed_));
}

template <typename T> bool Training::NNEngineTrainer<T>::init()
{
    if (save_nn_flag_)
    {
        try
        {
            if (!save_path_nn_.empty() && !std::filesystem::exists(save_path_nn_))
            {
                std::filesystem::create_directories(save_path_nn_);
            }
        } catch (const std::filesystem::filesystem_error& e)
        {
            std::cerr << "Error creating save directory " << save_path_nn_ << ": " << e.what() << std::endl;
            return false;
        }
    }

    net_ = std::make_unique<nn::ANN_MLP_GA<T>>(nn_size_config_, nn_seed_, static_cast<size_t>(population_size_config_),
                                               static_cast<size_t>(top_individuals_config_),
                                               static_cast<size_t>(activation_id_config_),
                                               static_cast<size_t>(elitism_config_));

    net_->SetName(nn_name_.c_str());

    net_->SetPopulationStrategy(nn::PopulationStrategy::MIXED_WITH_RANDOM_INJECTION, 0.30); // 30% random injection
    net_->CreatePopulation(true); // Initial population creation

    current_generation_ = 0;
    overall_start_time_ = std::chrono::steady_clock::now();

    return true;
}

template <typename T> bool Training::NNEngineTrainer<T>::load(const std::string& step)
{
    net_ = std::make_unique<nn::ANN_MLP_GA<T>>(); // Create empty network
    net_->SetName(nn_name_.c_str());

    std::string filename;
    if (step == "last") { filename = nn_name_ + "_last.txt"; }
    else
    {
        // Potentially more complex step parsing if needed, e.g. "gen_0010"
        filename = nn_name_ + "_" + step + ".txt";
    }

    std::filesystem::path full_path = std::filesystem::path(save_path_nn_) / filename;

    if (!std::filesystem::exists(full_path))
    {
        std::cerr << "Error loading network: File not found at " << full_path.string() << std::endl;
        net_.reset(); // Ensure net_ is null if load fails
        return false;
    }

    try
    {
        net_->Deserialize(full_path.string());
        current_generation_ = static_cast<int>(net_->GetEpochs());
        nn_size_config_     = net_->GetNetworkSize();           // Update size from loaded net
        overall_start_time_ = std::chrono::steady_clock::now(); // Reset timer

        return true;
    } catch (const std::exception& e)
    {
        std::cerr << "Error loading network from " << full_path.string() << ": " << e.what() << std::endl;
        net_.reset();
        return false;
    }
}

template <typename T> bool Training::NNEngineTrainer<T>::save()
{
    if (!net_)
    {
        std::cerr << "Save error: Network not initialized." << std::endl;
        return false;
    }

    std::string base_prefix = nn_name_;
    std::string filename_numbered;
    const auto& current_nn_size_vec = net_->GetNetworkSize(); // Use actual size from net

    if (!overwrite_save_)
    {
        std::string nn_size_str;
        for (size_t i = 0; i < current_nn_size_vec.size(); ++i)
        {
            nn_size_str += std::to_string(current_nn_size_vec[i]) + (i == current_nn_size_vec.size() - 1 ? "" : "_");
        }
        std::ostringstream ss_step;
        ss_step << std::setw(4) << std::setfill('0') << current_generation_;
        filename_numbered = base_prefix + "_" + nn_size_str + "_s_" + ss_step.str() + ".txt";
    }
    else { filename_numbered = base_prefix + "_latest_overwrite.txt"; }

    std::filesystem::path full_path_numbered = std::filesystem::path(save_path_nn_) / filename_numbered;
    std::filesystem::path full_path_last     = std::filesystem::path(save_path_nn_) / (nn_name_ + "_last.txt");

    try
    {
        if (std::filesystem::exists(full_path_numbered) && !overwrite_save_)
        { // Only remove if not overwriting and it's a numbered save
          // This logic might be tricky: if overwrite_save_ is false, we generate unique names.
          // If overwrite_save_ is true, filename_numbered is always "_latest_overwrite.txt"
          // The python code removes full_path if it exists before serializing. Let's mimic that for the numbered
          // file.
        }
        if (std::filesystem::exists(full_path_numbered)) { std::filesystem::remove(full_path_numbered); }

        net_->Serialize(full_path_numbered.string());

        if (std::filesystem::exists(full_path_last)) { std::filesystem::remove(full_path_last); }
        std::filesystem::copy_file(full_path_numbered, full_path_last,
                                   std::filesystem::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e)
    {
        std::cerr << "Error during network serialization/copy: " << e.what() << std::endl;
        return false;
    }
}

template <typename T> std::string Training::NNEngineTrainer<T>::format_time(long long total_seconds)
{
    if (total_seconds < 0) return "00:00";
    long long minutes = total_seconds / 60;
    long long seconds = total_seconds % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << minutes << ":" << std::setw(2) << std::setfill('0') << seconds;
    return oss.str();
}

template <typename T>
void Training::NNEngineTrainer<T>::balance_layouts(std::vector<LayoutInfo>& layouts_vec, int num_layouts_to_balance,
                                                   double target_ltr_ratio)
{
    if (target_ltr_ratio < 0.0 || target_ltr_ratio > 1.0) { target_ltr_ratio = 0.5; }

    int current_ltr_count = 0;
    for (const auto& layout : layouts_vec)
    {
        if (layout.spad_center < layout.lpad_center) { current_ltr_count++; }
    }

    int target_ltr_count_rounded =
        static_cast<int>(std::round(static_cast<double>(num_layouts_to_balance) * target_ltr_ratio));
    int num_to_flip = std::abs(current_ltr_count - target_ltr_count_rounded);

    if (num_to_flip == 0) { return; }

    bool flip_ltr_to_rtl   = current_ltr_count > target_ltr_count_rounded;
    double screen_center_x = cfg_width_ / 2.0;
    int flipped_count      = 0;

    for (int i = 0; i < num_layouts_to_balance && flipped_count < num_to_flip; ++i)
    {
        bool is_ltr = layouts_vec[static_cast<size_t>(i)].spad_center < layouts_vec[static_cast<size_t>(i)].lpad_center;
        if ((flip_ltr_to_rtl && is_ltr) || (!flip_ltr_to_rtl && !is_ltr))
        {
            // This layout is a candidate for flipping
            double old_spad_center    = layouts_vec[static_cast<size_t>(i)].spad_center;
            double old_lpad_center    = layouts_vec[static_cast<size_t>(i)].lpad_center;

            double new_spad_center    = 2.0 * screen_center_x - old_spad_center;
            double new_lpad_center    = 2.0 * screen_center_x - old_lpad_center;

            double new_spad_x1_double = new_spad_center - game_cfg_spad_width_ / 2.0;
            double new_lpad_x1_double = new_lpad_center - game_cfg_lpad_width_ / 2.0;

            // Ensure pads stay within bounds
            new_spad_x1_double        = std::max(0.0, std::min(new_spad_x1_double, cfg_width_ - game_cfg_spad_width_));
            new_lpad_x1_double        = std::max(0.0, std::min(new_lpad_x1_double, cfg_width_ - game_cfg_lpad_width_));

            layouts_vec[static_cast<size_t>(i)].spad_x1     = static_cast<int>(std::round(new_spad_x1_double));
            layouts_vec[static_cast<size_t>(i)].lpad_x1     = static_cast<int>(std::round(new_lpad_x1_double));
            layouts_vec[static_cast<size_t>(i)].spad_center = new_spad_center; // Update cached center
            layouts_vec[static_cast<size_t>(i)].lpad_center = new_lpad_center; // Update cached center

            flipped_count++;
        }
    }
    // Recalculate final counts for reporting
    current_ltr_count = 0;
    for (const auto& layout : layouts_vec)
    {
        if (layout.spad_center < layout.lpad_center) { current_ltr_count++; }
    }
}

template <typename T> void Training::NNEngineTrainer<T>::train()
{
    if (!net_)
    {
        std::cerr << "Train error: Network not initialized or loaded." << std::endl;
        return;
    }

    std::cout << "Starting C++ training for " << epochs_ << " generations..." << std::endl;
    std::cout << "Population size: " << net_->GetPopSize() << ", Top performers: " << net_->GetTopPerformersSize()
              << std::endl;
    std::cout << "Network structure: ";
    // Use nn_size_config_ as it's updated in load() after deserialization
    // to reflect the actual loaded network size.
    const auto& current_nn_size = nn_size_config_;
    for (size_t i = 0; i < current_nn_size.size(); ++i)
    {
        std::cout << current_nn_size[i] << (i == current_nn_size.size() - 1 ? "" : "-");
    }
    std::cout << std::endl;

    std::vector<LayoutInfo> layouts_vec;
    layouts_vec.reserve(static_cast<size_t>(layout_nb_));
    [[maybe_unused]] int initial_left_to_right_count = 0;

    unsigned int base_seed_for_layouts               = static_cast<unsigned int>(nn_seed_);

    for (int i = 0; i < layout_nb_; ++i)
    {
        // Use a deterministic seed for each layout if reproducibility per layout is desired
        std::optional<int> layout_seed_opt = static_cast<int>(base_seed_for_layouts + static_cast<unsigned int>(i));
        auto pad_pos = generate_random_pad_positions(cfg_width_, static_cast<int>(game_cfg_spad_width_),
                                                     static_cast<int>(game_cfg_lpad_width_), layout_seed_opt);

        LayoutInfo layout_item;
        layout_item.spad_x1     = pad_pos.first;
        layout_item.lpad_x1     = pad_pos.second;
        layout_item.spad_center = static_cast<double>(pad_pos.first) + game_cfg_spad_width_ / 2.0;
        layout_item.lpad_center = static_cast<double>(pad_pos.second) + game_cfg_lpad_width_ / 2.0;
        layouts_vec.push_back(layout_item);

        if (layout_item.spad_center < layout_item.lpad_center) { initial_left_to_right_count++; }
    }

    balance_layouts(layouts_vec, layout_nb_, left_right_ratio_);

    int final_ltr_count = 0;
    for (const auto& l : layouts_vec)
    {
        if (l.spad_center < l.lpad_center) final_ltr_count++;
    }
    std::cout << "Layouts after balancing: " << final_ltr_count << " left-to-right, " << (layout_nb_ - final_ltr_count)
              << " right-to-left." << std::endl;

    size_t population_size = net_->GetPopSize();
    std::vector<double> fitness_scores(population_size);
    std::vector<double> all_member_steps(population_size);

    for (int gen_offset = 0; gen_offset < epochs_; ++gen_offset)
    {
        auto gen_start_time    = std::chrono::steady_clock::now();
        int actual_gen_display = current_generation_ + 1;

        // Check for layout reset condition BEFORE train_generation for this epoch
        if (generations_stagnated_ >= reset_period_config_)
        {
            std::cout << "--- Stagnation detected. Resetting training layouts after " << generations_stagnated_
                      << " generations. --- " << std::endl;

            layouts_vec.clear();
            [[maybe_unused]] int initial_ltr_count_after_reset = 0;
            // Use a different seed base for new layouts to ensure variety
            unsigned int layout_reset_seed_base =
                static_cast<unsigned int>(nn_seed_ + current_generation_ + gen_offset);

            for (int i = 0; i < layout_nb_; ++i)
            {
                std::optional<int> layout_seed_opt =
                    static_cast<int>(layout_reset_seed_base + static_cast<unsigned int>(i));
                auto pad_pos = generate_random_pad_positions(cfg_width_, static_cast<int>(game_cfg_spad_width_),
                                                             static_cast<int>(game_cfg_lpad_width_), layout_seed_opt);
                LayoutInfo layout_item;
                layout_item.spad_x1     = pad_pos.first;
                layout_item.lpad_x1     = pad_pos.second;
                layout_item.spad_center = static_cast<double>(pad_pos.first) + game_cfg_spad_width_ / 2.0;
                layout_item.lpad_center = static_cast<double>(pad_pos.second) + game_cfg_lpad_width_ / 2.0;
                layouts_vec.push_back(layout_item);
                if (layout_item.spad_center < layout_item.lpad_center) { initial_ltr_count_after_reset++; }
            }
            balance_layouts(layouts_vec, layout_nb_, left_right_ratio_);

            generations_stagnated_     = 0;
            last_best_fitness_overall_ = std::numeric_limits<double>::max();
        }

        std::cout << "\n--- Generation " << actual_gen_display << " ---" << std::endl;

        train_generation(layouts_vec, population_size, fitness_scores, all_member_steps);

        std::vector<size_t> sorted_indices(population_size);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](size_t a, size_t b) { return fitness_scores[a] < fitness_scores[b]; });

        // Update stagnation counter based on the best performer of this generation
        if (!sorted_indices.empty() && population_size > 0)
        { // Ensure there are scores to check
            double current_best_fitness_this_gen = fitness_scores[sorted_indices[0]];
            const double epsilon                 = 1e-5; // Small tolerance for fitness improvement check

            // Check if member 0 is the best AND its fitness hasn't improved significantly
            if (sorted_indices[0] == 0 && current_best_fitness_this_gen >= (last_best_fitness_overall_ - epsilon))
            {
                generations_stagnated_++;
            }
            else
            {
                generations_stagnated_ = 0; // Reset if best is not member 0 or if fitness improved
            }
            last_best_fitness_overall_ = current_best_fitness_this_gen;
        }
        else
        {
            generations_stagnated_     = 0;
            last_best_fitness_overall_ = std::numeric_limits<double>::max();
        }

        net_->UpdateWeightsAndBiases(sorted_indices);
        net_->CreatePopulation(true);
        net_->UpdateEpochs(1);
        current_generation_     = static_cast<int>(net_->GetEpochs());

        auto gen_end_time       = std::chrono::steady_clock::now();
        auto gen_duration_s     = std::chrono::duration_cast<std::chrono::seconds>(gen_end_time - gen_start_time);
        auto total_elapsed_s    = std::chrono::duration_cast<std::chrono::seconds>(gen_end_time - overall_start_time_);

        double best_avg_fitness = fitness_scores[sorted_indices[0]] / static_cast<double>(layout_nb_);
        double mean_fitness_sum = 0;
        for (double score : fitness_scores) mean_fitness_sum += score;
        double mean_fitness =
            layout_nb_ > 0 ? (mean_fitness_sum / static_cast<double>(population_size)) / static_cast<double>(layout_nb_)
                           : 0.0;

        double mean_steps_sum = 0;
        for (double steps : all_member_steps) mean_steps_sum += steps;
        double mean_steps =
            layout_nb_ > 0 ? (mean_steps_sum / static_cast<double>(population_size)) / static_cast<double>(layout_nb_)
                           : 0.0;

        std::cout << "Generation " << current_generation_ << " complete. "
                  << "(Took: " << format_time(gen_duration_s.count()) << ", "
                  << "Total: " << format_time(total_elapsed_s.count()) << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Best Avg Fitness: " << best_avg_fitness << " (Member " << sorted_indices[0] << ")" << std::endl;
        std::cout << "  Avg Fitness:      " << mean_fitness << std::endl;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Avg Steps/Layout: " << mean_steps << std::endl;

        if (save_nn_flag_ && (current_generation_ % save_interval_ == 0 || gen_offset == epochs_ - 1)) { save(); }
    }
    std::cout << "\nC++ Training finished." << std::endl;
}

template <typename T>
void Training::NNEngineTrainer<T>::train_generation(
    const std::vector<LayoutInfo>& layouts, size_t population_size,
    std::vector<double>& fitness_scores,   // Output: per member total fitness
    std::vector<double>& all_member_steps) // Output: per member total steps
{
    // Initialize fitness scores and steps to zero for accumulation
    std::fill(fitness_scores.begin(), fitness_scores.end(), 0.0);
    std::fill(all_member_steps.begin(), all_member_steps.end(), 0.0);

    // Prepare config vectors once
    std::vector<double> x0_double = Config::getVectorDouble("GameCfg.x0");
    std::vector<T> x0_T(x0_double.size());
    std::transform(x0_double.begin(), x0_double.end(), x0_T.begin(), [](double val) { return static_cast<T>(val); });

    std::vector<double> v0_double = Config::getVectorDouble("GameCfg.v0");
    std::vector<T> v0_T(v0_double.size());
    std::transform(v0_double.begin(), v0_double.end(), v0_T.begin(), [](double val) { return static_cast<T>(val); });

    std::vector<double> a0_double = Config::getVectorDouble("GameCfg.a0");
    std::vector<T> a0_T(a0_double.size());
    std::transform(a0_double.begin(), a0_double.end(), a0_T.begin(), [](double val) { return static_cast<T>(val); });

    tp::thread_pool pool; // Thread pool for managing simulation tasks

    // Outer loop: Iterate over each member of the population
    for (size_t member_id = 0; member_id < population_size; ++member_id)
    {
        // Define a simulation task for the current member
        auto sim_task = [this, member_id, &layouts, &fitness_scores, &all_member_steps, &x0_T, &v0_T, &a0_T]() {
            // Initialize GameLogic and state buffer once per member task
            std::vector<T> state_buffer_local(5); // 5 state variables for the lander
            GameLogicCpp<T> game_sim(true);       // true for no_print_flag

            // Configure the game simulation for this member
            game_sim.set_config(static_cast<T>(this->cfg_width_), static_cast<T>(this->cfg_height_),
                                static_cast<T>(Config::getDouble("GameCfg.pad_y1", 50.0)),
                                static_cast<T>(Config::getDouble("GameCfg.terrain_y", 100.0)),
                                static_cast<T>(Config::getDouble("GameCfg.max_vx", 20.0)),
                                static_cast<T>(Config::getDouble("GameCfg.max_vy", 40.0)),
                                static_cast<T>(Config::getDouble("PlanetCfg.g", -1.62)),
                                static_cast<T>(Config::getDouble("PlanetCfg.mu_x", 0.1)),
                                static_cast<T>(Config::getDouble("PlanetCfg.mu_y", 0.1)),
                                static_cast<T>(Config::getDouble("LanderCfg.width", 20.0)),
                                static_cast<T>(Config::getDouble("LanderCfg.height", 20.0)),
                                static_cast<T>(Config::getInt("LanderCfg.max_fuel", 500)),
                                static_cast<T>(this->game_cfg_spad_width_), static_cast<T>(this->game_cfg_lpad_width_),
                                x0_T, v0_T, a0_T);

            double total_penalty_ltr_accum = 0.0;
            int count_layouts_ltr          = 0;
            int successful_landings_ltr    = 0;

            double total_penalty_rtl_accum = 0.0;
            int count_layouts_rtl          = 0;
            int successful_landings_rtl    = 0;

            double total_steps_this_member = 0.0;

            // Inner loop: Iterate over each layout for the current member
            for (const auto& layout_info : layouts)
            {
                bool is_ltr_layout = layout_info.spad_center < layout_info.lpad_center;
                game_sim.reset(static_cast<T>(layout_info.spad_x1), static_cast<T>(layout_info.lpad_x1));
                game_sim.get_state(state_buffer_local.data(), state_buffer_local.size());

                bool done_local                              = false;
                double accumulated_step_penalty_layout_local = 0.0;
                int steps_this_layout_local                  = 0;

                while (steps_this_layout_local < this->game_cfg_max_steps_)
                {
                    size_t action_idx =
                        this->net_->feedforward(state_buffer_local.data(), state_buffer_local.size(), member_id);

                    done_local = game_sim.update(static_cast<int>(action_idx), state_buffer_local.data(),
                                                 state_buffer_local.size());

                    accumulated_step_penalty_layout_local +=
                        static_cast<double>(game_sim.calculate_step_penalty(static_cast<int>(action_idx)));
                    steps_this_layout_local++;
                    if (done_local) { break; }
                } // End of simulation steps loop for one layout

                double terminal_penalty_layout_local =
                    static_cast<double>(game_sim.calculate_terminal_penalty(steps_this_layout_local));
                double penalty_for_layout = accumulated_step_penalty_layout_local + terminal_penalty_layout_local;

                total_steps_this_member += static_cast<double>(steps_this_layout_local);

                if (is_ltr_layout)
                {
                    total_penalty_ltr_accum += penalty_for_layout;
                    count_layouts_ltr++;
                    if (game_sim.landed_successfully) { successful_landings_ltr++; }
                }
                else
                {
                    total_penalty_rtl_accum += penalty_for_layout;
                    count_layouts_rtl++;
                    if (game_sim.landed_successfully) { successful_landings_rtl++; }
                }
            } // End of layouts loop for one member

            const double PENALTY_FOR_ZERO_SUCCESS_IN_DIRECTION = 50000.0;  // Increased penalty
            const double MAX_EXPECTED_PENALTY_NO_LAYOUTS       = 100000.0; // Fallback if a direction has no layouts

            double avg_penalty_ltr                             = MAX_EXPECTED_PENALTY_NO_LAYOUTS;
            if (count_layouts_ltr > 0)
            {
                avg_penalty_ltr = total_penalty_ltr_accum / static_cast<double>(count_layouts_ltr);
                if (successful_landings_ltr == 0)
                { // Penalize if LTR layouts were tried but none succeeded
                    avg_penalty_ltr += PENALTY_FOR_ZERO_SUCCESS_IN_DIRECTION;
                }
            }

            double avg_penalty_rtl = MAX_EXPECTED_PENALTY_NO_LAYOUTS;
            if (count_layouts_rtl > 0)
            {
                avg_penalty_rtl = total_penalty_rtl_accum / static_cast<double>(count_layouts_rtl);
                if (successful_landings_rtl == 0)
                { // Penalize if RTL layouts were tried but none succeeded
                    avg_penalty_rtl += PENALTY_FOR_ZERO_SUCCESS_IN_DIRECTION;
                }
            }

            double primary_fitness          = std::max(avg_penalty_ltr, avg_penalty_rtl);

            const double DUAL_LANDING_BONUS = 25000.0; // Increased bonus
            if (successful_landings_ltr > 0 && successful_landings_rtl > 0)
            {
                primary_fitness -= DUAL_LANDING_BONUS; // Lower penalty is better
            }

            fitness_scores[member_id]   = primary_fitness;
            all_member_steps[member_id] = total_steps_this_member;
        }; // End of sim_task lambda

        if (this->multithread_) { pool.push_task(sim_task); }
        else { sim_task(); } // Execute synchronously if multithreading is disabled
    } // End of population (member_id) loop

    // Wait for all member tasks to complete if multithreading is enabled
    if (this->multithread_) { pool.wait_for_tasks(); }
}

// Explicit template instantiation
template class Training::NNEngineTrainer<float>;
template class Training::NNEngineTrainer<double>;
