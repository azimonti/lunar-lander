/********************/
/*  main_train.cpp  */
/*   Version 1.0    */
/*    2023/05/08    */
/********************/
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "config_loader.h"
#include "nn_engine_trainer.h"

int main(int argc, char* argv[])
{
    // Load configuration from file first
    if (!Config::loadConfiguration("config.txt"))
    {
        std::cerr << "FATAL: Could not load configuration from config.txt. Exiting." << std::endl;
        return 1;
    }

    // Determine the type based on configuration
#if defined(USE_FLOAT)
    using TDataType = float;
    if (!Config::NNConfig::use_float)
    {
        std::cout << "Warning: Makefile defines USE_FLOAT, but config nn_config.use_float is False. Using float."
                  << std::endl;
    }
#else
    using TDataType = double;
    if (Config::NNConfig::use_float)
    {
        std::cout
            << "Warning: Makefile does not define USE_FLOAT, but config nn_config.use_float is True. Using double."
            << std::endl;
    }
#endif

    std::cout << "Using data type: " << (std::is_same_v<TDataType, float> ? "float" : "double") << std::endl;

    bool continue_training_flag    = false;
    std::string checkpoint_to_load = "last"; // Default for --continue without --step

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--continue") { continue_training_flag = true; }
        else if (arg == "--step")
        {
            if (i + 1 < argc)
            {
                try
                {
                    int step_val = std::stoi(argv[++i]); // Increment i to consume value
                    std::ostringstream oss;
                    // Format step_val to "gen_XXXX" as expected by NNEngineTrainer::load
                    oss << "gen_" << std::setw(4) << std::setfill('0') << step_val;
                    checkpoint_to_load     = oss.str();
                    // If --step is provided, we imply --continue
                    continue_training_flag = true;
                } catch (const std::invalid_argument& /*ia*/)
                {
                    std::cerr << "Error: Invalid argument for --step: " << argv[i] << " is not a valid integer."
                              << std::endl;
                    return 1;
                } catch (const std::out_of_range& /*oor*/)
                {
                    std::cerr << "Error: Argument for --step out of range: " << argv[i] << std::endl;
                    return 1;
                }
            }
            else
            {
                std::cerr << "Error: --step option requires an integer argument." << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Warning: Unknown argument: " << arg << std::endl;
            // You could print usage instructions here if desired
        }
    }

    Training::NNEngineTrainer<TDataType> trainer;

    // Initialize the trainer (creates the network or prepares for loading)
    if (!trainer.init())
    {
        std::cerr << "Failed to initialize NNEngineTrainer." << std::endl;
        return 1;
    }

    if (continue_training_flag)
    {
        std::cout << "Attempting to load network from checkpoint: " << checkpoint_to_load << std::endl;
        if (!trainer.load(checkpoint_to_load))
        {
            std::cerr << "Failed to load network from checkpoint '" << checkpoint_to_load
                      << "'. Training will start from scratch (using the newly initialized network)." << std::endl;
            // trainer.init() was already called, so a fresh network exists.
        }
        else
        {
            std::cout << "Successfully loaded network from checkpoint '" << checkpoint_to_load << "'." << std::endl;
        }
    }
    else
    {
        std::cout << "Starting training from scratch (no --continue specified, or --step without --continue)."
                  << std::endl;
    }

    // Start the training process
    trainer.train();

    std::cout << "Training process finished in C++." << std::endl;
    return 0;
}
