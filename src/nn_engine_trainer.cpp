/*************************/
/* nn_engine_trainer.cpp */
/*      Version 1.2      */
/*       2023/05/13      */
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
    nn_name_                = Config::getString("NNConfig.name");
    save_path_nn_           = Config::getString("NNConfig.save_path_nn");
    save_nn_flag_           = Config::getBool("NNConfig.save_nn");
    overwrite_save_         = Config::getBool("NNConfig.overwrite");
    save_interval_          = Config::getInt("NNConfig.save_interval");
    epochs_                 = Config::getInt("NNConfig.epochs");
    layout_nb_              = Config::getInt("NNTrainingCfg.layout_nb");
    left_right_ratio_       = Config::getDouble("NNTrainingCfg.left_right_ratio");

    nn_seed_                = Config::getInt("NNConfig.seed");
    population_size_config_ = Config::getInt("NNConfig.population_size");
    top_individuals_config_ = Config::getInt("NNConfig.top_individuals");
    activation_id_config_   = Config::getInt("NNConfig.activation_id");
    elitism_config_         = Config::getBool("NNConfig.elitism");
    multithread_            = Config::getBool("NNTrainingCfg.multithread");

    nn_size_config_.push_back(5); // Input layer: 5 state variables
    std::vector<int> hlayers_vec = Config::getVectorInt("NNConfig.hlayers");
    for (int h_size : hlayers_vec) { nn_size_config_.push_back(static_cast<size_t>(h_size)); }
    nn_size_config_.push_back(4); // Output layer: 4 actions

    cfg_width_                   = Config::getDouble("DisplayCfg.width");
    cfg_height_                  = Config::getDouble("DisplayCfg.height");
    game_cfg_spad_width_         = Config::getDouble("GameCfg.spad_width");
    game_cfg_lpad_width_         = Config::getDouble("GameCfg.lpad_width");
    game_cfg_max_steps_          = Config::getInt("GameCfg.max_steps");

    // Initialize stagnation tracking variables
    reset_period_config_         = Config::getInt("NNTrainingCfg.reset_period");
    generations_stagnated_       = 0;
    last_best_fitness_overall_   = std::numeric_limits<double>::max();

    // Load fitness function parameters
    random_injection_ratio_      = Config::getDouble("NNConfig.random_injection_ratio");

    // Fitness display
    nn_train_tp_landed_lr_bonus_ = static_cast<T>(Config::getDouble("NNTrainingCfg.tp_landed_lr_bonus"));

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
                                               static_cast<size_t>(activation_id_config_));

    net_->SetName(nn_name_.c_str());

    net_->SetPopulationStrategy(nn::PopulationStrategy::MIXED_WITH_RANDOM_INJECTION, random_injection_ratio_);
    net_->CreatePopulation(elitism_config_); // Initial population creation

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

// Updated signature and logic to populate layout directions
template <typename T>
void Training::NNEngineTrainer<T>::balance_layouts(std::vector<LayoutInfo>& layouts_vec,
                                                   std::vector<size_t>& layout_directions, int num_layouts_to_balance,
                                                   double target_ltr_ratio)
{
    if (target_ltr_ratio < 0.0 || target_ltr_ratio > 1.0) { target_ltr_ratio = 0.5; }

    // The vector is already sized from its declaration, no need to clear or resize.
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
            layouts_vec[static_cast<size_t>(i)].spad_center = new_spad_center;
            layouts_vec[static_cast<size_t>(i)].lpad_center = new_lpad_center;

            flipped_count++;
        }
    }
    // Recalculate final counts for reporting
    current_ltr_count = 0;
    for (const auto& layout : layouts_vec)
    {
        if (layout.spad_center < layout.lpad_center) { current_ltr_count++; }
    }

    // After balancing (or even if no balancing occurs), determine and assign the final direction for each layout
    for (size_t i = 0; i < static_cast<size_t>(num_layouts_to_balance); ++i)
    {
        layout_directions[i] =
            (layouts_vec[i].spad_center < layouts_vec[i].lpad_center) ? 0 : 1; // 0 for LTR, 1 for RTL
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
              << ", Elitism: " << elitism_config_ << std::endl;
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
    // Declare and initialize vector for layout directions with its size
    std::vector<size_t> layout_directions(static_cast<size_t>(layout_nb_));
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

    // Pass layout_directions to balance_layouts
    balance_layouts(layouts_vec, layout_directions, layout_nb_, left_right_ratio_);

    int final_ltr_count = 0;
    for (const auto& l : layouts_vec)
    {
        if (l.spad_center < l.lpad_center) final_ltr_count++;
    }
    std::cout << "Layouts after balancing: " << final_ltr_count << " left-to-right, " << (layout_nb_ - final_ltr_count)
              << " right-to-left." << std::endl;

    size_t population_size = net_->GetPopSize();
    std::vector<double> fitness_scores(population_size);
    std::vector<double> last_gen_lr(population_size);
    std::vector<double> all_member_steps(population_size);

    // Pre-allocate game simulators
    std::vector<GameLogicCpp<T>> game_sims;
    game_sims.reserve(population_size); // Reserve space
    for (size_t i = 0; i < population_size; ++i)
    {
        game_sims.emplace_back(true); // Construct in-place with no_print_flag = true
    }

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
            // Pass layout_directions to balance_layouts during reset as well
            balance_layouts(layouts_vec, layout_directions, layout_nb_, left_right_ratio_);

            generations_stagnated_     = 0;
            last_best_fitness_overall_ = std::numeric_limits<double>::max();
        }

        std::cout << "\n--- Generation " << actual_gen_display << " ---" << std::endl;

        // Pass layout_directions to train_generation
        train_generation(layouts_vec, layout_directions, population_size, fitness_scores, last_gen_lr, all_member_steps,
                         game_sims);

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
        net_->CreatePopulation(elitism_config_);
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
        std::cout << "  Min Number of Landing in Both Directions: "
                  << calculate_combined_landing_nb(last_gen_lr[sorted_indices[0]]) << std::endl;
        std::cout << "  Avg Fitness:      " << mean_fitness << std::endl;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Avg Steps/Layout: " << mean_steps << std::endl;

        if (save_nn_flag_ && (current_generation_ % save_interval_ == 0 || gen_offset == epochs_ - 1)) { save(); }
    }
    std::cout << "\nC++ Training finished." << std::endl;
}

// Updated signature
template <typename T>
void Training::NNEngineTrainer<T>::train_generation(const std::vector<LayoutInfo>& layouts,
                                                    const std::vector<size_t>& layout_directions, // Added
                                                    size_t population_size, std::vector<double>& fitness_scores,
                                                    std::vector<double>& last_gen_lr,
                                                    std::vector<double>& all_member_steps,
                                                    std::vector<GameLogicCpp<T>>& game_sims)
{
    // Initialize fitness scores and steps to zero for accumulation
    std::fill(fitness_scores.begin(), fitness_scores.end(), 0.0);
    std::fill(last_gen_lr.begin(), last_gen_lr.end(), 0.0);
    std::fill(all_member_steps.begin(), all_member_steps.end(), 0.0);

    // Vector to store potential left/right landing bonuses for each member
    // Declared outside the parallel loop to avoid repeated allocation
    std::vector<std::array<T, 2>> landing_bonuses(population_size); // Holds potential bonus values

    tp::thread_pool pool; // Thread pool for managing simulation tasks

    // Outer loop: Iterate over each member of the population
    for (size_t member_id = 0; member_id < population_size; ++member_id)
    {
        // Define a simulation task for the current member
        auto sim_task = [this, member_id, &layouts, &layout_directions, &fitness_scores, &last_gen_lr,
                         &all_member_steps, &game_sims, &landing_bonuses]() {
            // Initialize state buffer once per member task
            std::vector<T> state_buffer_local(5);             // 5 state variables for the lander
            GameLogicCpp<T>& game_sim = game_sims[member_id]; // Use pre-allocated simulator
            // Get reference to this member's landing bonus array and initialize it
            std::array<T, 2>& member_landing_bonus_lr =
                landing_bonuses[member_id];                        // Reference to the specific member's array
            member_landing_bonus_lr[0]                   = T(0.0); // Initialize potential LTR bonus
            member_landing_bonus_lr[1]                   = T(0.0); // Initialize potential RTL bonus

            double total_fitness_for_member              = 0.0;
            double total_steps_for_member                = 0.0;
            bool done_local                              = false;
            double accumulated_step_penalty_layout_local = 0.0;
            double fitness_for_layout                    = 0.0;
            int steps_this_layout_local                  = 0;
            size_t action_idx                            = 0;

            // Inner loop: Iterate over each layout for the current member
            for (size_t layout_idx = 0; layout_idx < layouts.size(); ++layout_idx) // Use index for direction lookup
            {
                const auto& layout_info  = layouts[layout_idx];
                size_t current_direction = layout_directions[layout_idx]; // Get direction for this layout

                game_sim.reset(static_cast<T>(layout_info.spad_x1), static_cast<T>(layout_info.lpad_x1));
                game_sim.get_state(state_buffer_local.data(), state_buffer_local.size());

                done_local                            = false;
                accumulated_step_penalty_layout_local = 0.0;
                steps_this_layout_local               = 0;
                action_idx                            = 0;

                while (steps_this_layout_local < this->game_cfg_max_steps_)
                {
                    action_idx =
                        this->net_->feedforward(state_buffer_local.data(), state_buffer_local.size(), member_id);

                    done_local = game_sim.update(static_cast<int>(action_idx), state_buffer_local.data(),
                                                 state_buffer_local.size());

                    accumulated_step_penalty_layout_local +=
                        static_cast<double>(game_sim.calculate_step_penalty(static_cast<int>(action_idx)));
                    steps_this_layout_local++;
                    if (done_local) { break; }
                } // End of simulation steps loop for one layout

                // Calculate terminal penalty
                fitness_for_layout = static_cast<double>(game_sim.calculate_terminal_penalty(
                    steps_this_layout_local, current_direction, member_landing_bonus_lr));

                fitness_for_layout += accumulated_step_penalty_layout_local;
                total_fitness_for_member += fitness_for_layout;
                total_steps_for_member += static_cast<double>(steps_this_layout_local);
            } // End of layouts loop for one member
            last_gen_lr[member_id] =
                static_cast<double>(std::min(member_landing_bonus_lr[0], member_landing_bonus_lr[1]));

            // Apply the final L/R bonus by subtracting the minimum of the accumulated directional bonuses
            total_fitness_for_member -= last_gen_lr[member_id];

            // Store the final accumulated fitness and steps for this member
            // This is thread-safe as each task writes to a unique member_id index.
            fitness_scores[member_id]   = total_fitness_for_member;
            all_member_steps[member_id] = total_steps_for_member;
        }; // End of sim_task lambda

        if (this->multithread_) { pool.push_task(sim_task); }
        else { sim_task(); } // Execute synchronously if multithreading is disabled
    } // End of population (member_id) loop

    // Wait for all member tasks to complete if multithreading is enabled
    if (this->multithread_) { pool.wait_for_tasks(); }
}

template <typename T>
int Training::NNEngineTrainer<T>::calculate_combined_landing_nb(double total_landing_bonus_lr) const
{
    return static_cast<int>(std::round(total_landing_bonus_lr / nn_train_tp_landed_lr_bonus_));
}

// Explicit template instantiation
template class Training::NNEngineTrainer<float>;
template class Training::NNEngineTrainer<double>;
