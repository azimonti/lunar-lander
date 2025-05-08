/*************************/
/* nn_engine_trainer.cpp */
/*      Version 1.0      */
/*       2023/05/08      */
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
#include "config_cpp.h"
#include "game_logic.h"
#include "game_utils.h"
#include "nn_engine_trainer.h"

template <typename T>
Training::NNEngineTrainer<T>::NNEngineTrainer(bool verbose_override)
    : current_generation_(0), verbose_flag_(verbose_override || Config::NNConfig::verbose)
{
    // Cache configuration values
    nn_name_                 = Config::NNConfig::name;
    save_path_nn_            = Config::NNConfig::save_path_nn;
    save_nn_flag_            = Config::NNConfig::save_nn;
    overwrite_save_          = Config::NNConfig::overwrite;
    save_interval_           = Config::NNConfig::save_interval;
    epochs_                  = Config::NNConfig::epochs;
    layout_nb_               = Config::NNConfig::layout_nb;
    left_right_ratio_        = Config::NNConfig::left_right_ratio;

    nn_seed_                 = Config::NNConfig::seed;
    population_size_config_  = Config::NNConfig::population_size;
    top_individuals_config_  = Config::NNConfig::top_individuals;
    activation_id_config_    = Config::NNConfig::activation_id;
    elitism_config_          = Config::NNConfig::elitism;
    mixed_population_config_ = Config::NNConfig::mixed_population;
    multithread_             = Config::NNConfig::multithread;

    nn_size_config_.push_back(5); // Input layer: 5 state variables
    for (int h_size : Config::NNConfig::hlayers) { nn_size_config_.push_back(static_cast<size_t>(h_size)); }
    nn_size_config_.push_back(4); // Output layer: 4 actions

    cfg_width_           = Config::Cfg::width;
    cfg_height_          = Config::Cfg::height;
    game_cfg_spad_width_ = Config::GameCfg::spad_width;
    game_cfg_lpad_width_ = Config::GameCfg::lpad_width;
    game_cfg_max_steps_  = Config::GameCfg::max_steps;

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
    net_->SetMixed(mixed_population_config_);
    net_->CreatePopulation(true);

    current_generation_ = 0;
    overall_start_time_ = std::chrono::steady_clock::now();
    if (verbose_flag_)
    {
        std::cout << "NNEngineTrainer initialized." << std::endl;
        std::cout << "  Network Name: " << nn_name_ << std::endl;
        std::cout << "  Save Path: " << save_path_nn_ << std::endl;
        std::cout << "  NN Structure: ";
        for (size_t i = 0; i < nn_size_config_.size(); ++i)
        {
            std::cout << nn_size_config_[i] << (i == nn_size_config_.size() - 1 ? "" : "-");
        }
        std::cout << std::endl;
    }
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
        if (verbose_flag_)
        {
            std::cout << "Network loaded from: " << full_path.string() << " (Generation " << current_generation_ << ")"
                      << std::endl;
            std::cout << "  Network structure from file: ";
            for (size_t i = 0; i < nn_size_config_.size(); ++i)
            {
                std::cout << nn_size_config_[i] << (i == nn_size_config_.size() - 1 ? "" : "-");
            }
            std::cout << std::endl;
        }
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
        if (verbose_flag_) { std::cout << "Network serialized to: " << full_path_numbered.string() << std::endl; }

        if (std::filesystem::exists(full_path_last)) { std::filesystem::remove(full_path_last); }
        std::filesystem::copy_file(full_path_numbered, full_path_last,
                                   std::filesystem::copy_options::overwrite_existing);
        if (verbose_flag_) { std::cout << "Copied network state to: " << full_path_last.string() << std::endl; }
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
    if (target_ltr_ratio < 0.0 || target_ltr_ratio > 1.0)
    {
        if (verbose_flag_)
        {
            std::cout << "  Warning: Invalid left_right_ratio (" << target_ltr_ratio << "). Defaulting to 0.5."
                      << std::endl;
        }
        target_ltr_ratio = 0.5;
    }

    int current_ltr_count = 0;
    for (const auto& layout : layouts_vec)
    {
        if (layout.spad_center < layout.lpad_center) { current_ltr_count++; }
    }

    int target_ltr_count_rounded =
        static_cast<int>(std::round(static_cast<double>(num_layouts_to_balance) * target_ltr_ratio));
    int num_to_flip = std::abs(current_ltr_count - target_ltr_count_rounded);

    if (num_to_flip == 0)
    {
        if (verbose_flag_)
        {
            std::cout << "Layouts already meet target ratio (" << target_ltr_ratio << "): " << current_ltr_count
                      << " left-to-right, " << (num_layouts_to_balance - current_ltr_count) << " right-to-left."
                      << std::endl;
        }
        return;
    }

    bool flip_ltr_to_rtl = current_ltr_count > target_ltr_count_rounded;
    if (verbose_flag_)
    {
        std::cout << "  Balancing (Ratio " << target_ltr_ratio << "): Flipping " << num_to_flip
                  << (flip_ltr_to_rtl ? " left-to-right" : " right-to-left") << " layouts to reach target "
                  << target_ltr_count_rounded << "." << std::endl;
    }

    double screen_center_x = cfg_width_ / 2.0;
    int flipped_count      = 0;

    for (int i = 0; i < num_layouts_to_balance && flipped_count < num_to_flip; ++i)
    {
        bool is_ltr = layouts_vec[static_cast<size_t>(i)].spad_center < layouts_vec[static_cast<size_t>(i)].lpad_center;
        if ((flip_ltr_to_rtl && is_ltr) || (!flip_ltr_to_rtl && !is_ltr))
        {
            // This layout is a candidate for flipping
            double old_spad_x1        = static_cast<double>(layouts_vec[static_cast<size_t>(i)].spad_x1);
            double old_lpad_x1        = static_cast<double>(layouts_vec[static_cast<size_t>(i)].lpad_x1);
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

            if (verbose_flag_)
            {
                std::cout << "    Flipped layout " << i + 1 << ": "
                          << "Old Start=" << std::fixed << std::setprecision(1) << old_spad_x1
                          << ", Old Landing=" << old_lpad_x1 << " -> "
                          << "New Start=" << layouts_vec[static_cast<size_t>(i)].spad_x1
                          << ", New Landing=" << layouts_vec[static_cast<size_t>(i)].lpad_x1 << std::endl;
            }
            flipped_count++;
        }
    }
    // Recalculate final counts for reporting
    current_ltr_count = 0;
    for (const auto& layout : layouts_vec)
    {
        if (layout.spad_center < layout.lpad_center) { current_ltr_count++; }
    }
    if (verbose_flag_)
    {
        double final_ratio =
            num_layouts_to_balance > 0 ? static_cast<double>(current_ltr_count) / num_layouts_to_balance : 0.0;
        std::cout << "Layouts after ratio balancing: " << current_ltr_count << " left-to-right (" << std::fixed
                  << std::setprecision(1) << (final_ratio * 100.0) << "%), "
                  << (num_layouts_to_balance - current_ltr_count) << " right-to-left." << std::endl;
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
    const auto& current_nn_size = net_->GetNetworkSize();
    for (size_t i = 0; i < current_nn_size.size(); ++i)
    {
        std::cout << current_nn_size[i] << (i == current_nn_size.size() - 1 ? "" : "-");
    }
    std::cout << std::endl;

    std::vector<LayoutInfo> layouts_vec;
    layouts_vec.reserve(static_cast<size_t>(layout_nb_));
    int initial_left_to_right_count = 0;

    if (verbose_flag_) { std::cout << "Generating " << layout_nb_ << " layouts for training..." << std::endl; }
    unsigned int base_seed_for_layouts = static_cast<unsigned int>(nn_seed_); // Or use a different seed source

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
        if (verbose_flag_)
        {
            std::cout << "  Layout " << i + 1 << ": Start=" << pad_pos.first << ", Landing=" << pad_pos.second
                      << std::endl;
        }
    }
    if (verbose_flag_)
    {
        std::cout << "Initial layouts: " << initial_left_to_right_count << " left-to-right, "
                  << (layout_nb_ - initial_left_to_right_count) << " right-to-left." << std::endl;
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

        std::cout << "\n--- Generation " << actual_gen_display << " ---" << std::endl;

        train_generation(layouts_vec, population_size, fitness_scores, all_member_steps);

        std::vector<size_t> sorted_indices(population_size);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](size_t a, size_t b) { return fitness_scores[a] < fitness_scores[b]; });

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
    std::vector<T> x0_T(Config::GameCfg::x0.size());
    std::transform(Config::GameCfg::x0.begin(), Config::GameCfg::x0.end(), x0_T.begin(),
                   [](double val) { return static_cast<T>(val); });
    std::vector<T> v0_T(Config::GameCfg::v0.size());
    std::transform(Config::GameCfg::v0.begin(), Config::GameCfg::v0.end(), v0_T.begin(),
                   [](double val) { return static_cast<T>(val); });
    std::vector<T> a0_T(Config::GameCfg::a0.size());
    std::transform(Config::GameCfg::a0.begin(), Config::GameCfg::a0.end(), a0_T.begin(),
                   [](double val) { return static_cast<T>(val); });

    tp::thread_pool pool;

    for (const auto& layout_info : layouts)
    {
        for (size_t member_id = 0; member_id < population_size; ++member_id)
        {
            auto sim_task = [this, &layout_info, member_id, &fitness_scores, &all_member_steps, &x0_T, &v0_T, &a0_T]() {
                std::vector<T> state_buffer_local(5); // 5 state variables for the lander, local to task
                GameLogicCpp<T> game_sim(true);       // true for no_print_flag, local to task

                game_sim.set_config(static_cast<T>(this->cfg_width_), static_cast<T>(this->cfg_height_),
                                    static_cast<T>(Config::GameCfg::pad_y1), static_cast<T>(Config::GameCfg::terrain_y),
                                    static_cast<T>(Config::GameCfg::max_vx), static_cast<T>(Config::GameCfg::max_vy),
                                    static_cast<T>(Config::PlanetCfg::g), static_cast<T>(Config::PlanetCfg::mu_x),
                                    static_cast<T>(Config::PlanetCfg::mu_y), static_cast<T>(Config::LanderCfg::width),
                                    static_cast<T>(Config::LanderCfg::height),
                                    static_cast<T>(Config::LanderCfg::max_fuel),
                                    static_cast<T>(this->game_cfg_spad_width_),
                                    static_cast<T>(this->game_cfg_lpad_width_), x0_T, v0_T, a0_T);

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
                }
                double terminal_penalty_layout_local =
                    static_cast<double>(game_sim.calculate_terminal_penalty(steps_this_layout_local));
                double fitness_for_layout = accumulated_step_penalty_layout_local + terminal_penalty_layout_local;

                // Accumulate fitness and steps for this member from this layout
                // This is safe because each task updates a unique fitness_scores[member_id] /
                // all_member_steps[member_id] for the current layout, and pool.wait_for_tasks() synchronizes before the
                // next layout.
                fitness_scores[member_id] += fitness_for_layout;
                all_member_steps[member_id] += static_cast<double>(steps_this_layout_local);
            };

            if (this->multithread_) { pool.push_task(sim_task); }
            else { sim_task(); }
        } // End of population loop

        if (this->multithread_) { pool.wait_for_tasks(); } // Wait for all members to finish for the current layout
    } // End of layouts loop
}

// Explicit template instantiation
template class Training::NNEngineTrainer<float>;
template class Training::NNEngineTrainer<double>;
