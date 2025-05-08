/*********************/
/* config_loader.cpp */
/*    Version 1.0    */
/*     2023/05/09    */
/*********************/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "config_loader.h"

// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first) { return str; }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

// Helper function to split a string by a delimiter
std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) { tokens.push_back(trim(token)); }
    return tokens;
}

namespace Config
{

    namespace LanderCfg
    {
        double width;
        double height;
        int max_fuel;
    } // namespace LanderCfg

    namespace Cfg
    {
        double width;
        double height;
        int fps;
        bool save_img;
        std::string save_path_img;
        bool with_sounds;
        bool verbose;
    } // namespace Cfg

    namespace GameCfg
    {
        bool random_position;
        std::vector<double> x0;
        std::vector<double> v0;
        std::vector<double> a0;
        double spad_x1;
        double spad_width;
        double lpad_x1;
        double lpad_width;
        double pad_y1;
        double pad_height;
        double terrain_y;
        double max_vx;
        double max_vy;
        int max_steps;
        int current_seed;
    } // namespace GameCfg

    namespace PlanetCfg
    {
        double g;
        double mu_x;
        double mu_y;
    } // namespace PlanetCfg

    namespace NNConfig
    {
        std::string name;
        std::vector<int> hlayers;
        bool use_float;
        int seed;
        int top_individuals;
        int population_size;
        bool mixed_population;
        bool elitism;
        int activation_id;
        bool save_nn;
        bool overwrite;
        std::string save_path_nn;
        int save_interval;
        int epochs;
        int layout_nb;
        double left_right_ratio;
        bool multithread;
        bool verbose;
    } // namespace NNConfig

    // Function to load configuration from a file
    bool loadConfiguration(const std::string& filepath)
    {
        std::ifstream configFile(filepath, std::ios::binary);
        if (!configFile.is_open())
        {
            std::cerr << "Error: Could not open configuration file: " << filepath << std::endl;
            return false;
        }

        std::string line;
        std::map<std::string, std::string> rawConfig;

        while (std::getline(configFile, line))
        {
            line = trim(line);
            if (line.empty() || line[0] == '#')
            { // Skip empty lines and comments
                continue;
            }

            size_t delimiterPos = line.find('=');
            if (delimiterPos == std::string::npos)
            {
                std::cerr << "Warning: Malformed line in config (missing '='): " << line << std::endl;
                continue;
            }

            std::string key   = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));
            rawConfig[key]    = value;
        }
        configFile.close();

        try
        {
            auto get_bool = [&](const std::string& val_str) {
                std::string lower_val = val_str;
                std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(), ::tolower);
                if (lower_val == "true") return true;
                if (lower_val == "false") return false;
                throw std::invalid_argument("Invalid boolean value: " + val_str);
            };

            auto get_vec_double = [&](const std::string& val_str) {
                std::vector<double> vec;
                std::vector<std::string> parts = split(val_str, ',');
                for (const auto& p : parts) vec.push_back(std::stod(p));
                return vec;
            };

            auto get_vec_int = [&](const std::string& val_str) {
                std::vector<int> vec;
                std::vector<std::string> parts = split(val_str, ',');
                for (const auto& p : parts) vec.push_back(std::stoi(p));
                return vec;
            };

            // LanderCfg
            LanderCfg::width           = std::stod(rawConfig.at("LanderCfg.width"));
            LanderCfg::height          = std::stod(rawConfig.at("LanderCfg.height"));
            LanderCfg::max_fuel        = std::stoi(rawConfig.at("LanderCfg.max_fuel"));

            // Cfg
            Cfg::width                 = std::stod(rawConfig.at("Cfg.width"));
            Cfg::height                = std::stod(rawConfig.at("Cfg.height"));
            Cfg::fps                   = std::stoi(rawConfig.at("Cfg.fps"));
            Cfg::save_img              = get_bool(rawConfig.at("Cfg.save_img"));
            Cfg::save_path_img         = rawConfig.at("Cfg.save_path_img");
            Cfg::with_sounds           = get_bool(rawConfig.at("Cfg.with_sounds"));
            Cfg::verbose               = get_bool(rawConfig.at("Cfg.verbose"));

            // GameCfg
            GameCfg::random_position   = get_bool(rawConfig.at("GameCfg.random_position"));
            GameCfg::x0                = get_vec_double(rawConfig.at("GameCfg.x0"));
            GameCfg::v0                = get_vec_double(rawConfig.at("GameCfg.v0"));
            GameCfg::a0                = get_vec_double(rawConfig.at("GameCfg.a0"));
            GameCfg::spad_x1           = std::stod(rawConfig.at("GameCfg.spad_x1"));
            GameCfg::spad_width        = std::stod(rawConfig.at("GameCfg.spad_width"));
            GameCfg::lpad_x1           = std::stod(rawConfig.at("GameCfg.lpad_x1"));
            GameCfg::lpad_width        = std::stod(rawConfig.at("GameCfg.lpad_width"));
            GameCfg::pad_y1            = std::stod(rawConfig.at("GameCfg.pad_y1"));
            GameCfg::pad_height        = std::stod(rawConfig.at("GameCfg.pad_height"));
            GameCfg::terrain_y         = std::stod(rawConfig.at("GameCfg.terrain_y"));
            GameCfg::max_vx            = std::stod(rawConfig.at("GameCfg.max_vx"));
            GameCfg::max_vy            = std::stod(rawConfig.at("GameCfg.max_vy"));
            GameCfg::max_steps         = std::stoi(rawConfig.at("GameCfg.max_steps"));
            GameCfg::current_seed      = std::stoi(rawConfig.at("GameCfg.current_seed"));

            // PlanetCfg
            PlanetCfg::g               = std::stod(rawConfig.at("PlanetCfg.g"));
            PlanetCfg::mu_x            = std::stod(rawConfig.at("PlanetCfg.mu_x"));
            PlanetCfg::mu_y            = std::stod(rawConfig.at("PlanetCfg.mu_y"));

            // NNConfig
            NNConfig::name             = rawConfig.at("NNConfig.name");
            NNConfig::hlayers          = get_vec_int(rawConfig.at("NNConfig.hlayers"));
            NNConfig::use_float        = get_bool(rawConfig.at("NNConfig.use_float"));
            NNConfig::seed             = std::stoi(rawConfig.at("NNConfig.seed"));
            NNConfig::top_individuals  = std::stoi(rawConfig.at("NNConfig.top_individuals"));
            NNConfig::population_size  = std::stoi(rawConfig.at("NNConfig.population_size"));
            NNConfig::mixed_population = get_bool(rawConfig.at("NNConfig.mixed_population"));
            NNConfig::elitism          = get_bool(rawConfig.at("NNConfig.elitism"));
            NNConfig::activation_id    = std::stoi(rawConfig.at("NNConfig.activation_id"));
            NNConfig::save_nn          = get_bool(rawConfig.at("NNConfig.save_nn"));
            NNConfig::overwrite        = get_bool(rawConfig.at("NNConfig.overwrite"));
            NNConfig::save_path_nn     = rawConfig.at("NNConfig.save_path_nn");
            NNConfig::save_interval    = std::stoi(rawConfig.at("NNConfig.save_interval"));
            NNConfig::epochs           = std::stoi(rawConfig.at("NNConfig.epochs"));
            NNConfig::layout_nb        = std::stoi(rawConfig.at("NNConfig.layout_nb"));
            NNConfig::left_right_ratio = std::stod(rawConfig.at("NNConfig.left_right_ratio"));
            NNConfig::multithread      = get_bool(rawConfig.at("NNConfig.multithread"));
            NNConfig::verbose          = get_bool(rawConfig.at("NNConfig.verbose"));

        } catch (const std::out_of_range& oor)
        {
            std::cerr << "Error: Configuration key not found or out of range: " << oor.what() << std::endl;
            return false;
        } catch (const std::invalid_argument& ia)
        {
            std::cerr << "Error: Invalid argument in configuration value: " << ia.what() << std::endl;
            return false;
        } catch (const std::exception& e)
        {
            std::cerr << "An unexpected error occurred during configuration parsing: " << e.what() << std::endl;
            return false;
        }

        std::cout << "Configuration successfully loaded from " << filepath << std::endl;
        return true;
    }

} // namespace Config
