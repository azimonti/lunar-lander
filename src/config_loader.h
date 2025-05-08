#ifndef _CONFIG_LOADER_H_303ecb31ae304863961708f4a1fa3410_
#define _CONFIG_LOADER_H_303ecb31ae304863961708f4a1fa3410_

/*******************/
/* config_loader.h */
/*   Version 1.0   */
/*    2023/05/09   */
/*******************/

#include <array>
#include <map>
#include <string>
#include <vector>

namespace Config
{

    namespace LanderCfg
    {
        extern double width;
        extern double height;
        extern int max_fuel;
    } // namespace LanderCfg

    namespace Cfg
    {
        extern double width;
        extern double height;
        extern int fps;
        extern bool save_img;
        extern std::string save_path_img;
        extern bool with_sounds;
        extern bool verbose;
    } // namespace Cfg

    namespace GameCfg
    {
        extern bool random_position;
        extern std::vector<double> x0;
        extern std::vector<double> v0;
        extern std::vector<double> a0;
        extern double spad_x1;
        extern double spad_width;
        extern double lpad_x1;
        extern double lpad_width;
        extern double pad_y1;
        extern double pad_height;
        extern double terrain_y;
        extern double max_vx;
        extern double max_vy;
        extern int max_steps;
        extern int current_seed;
    } // namespace GameCfg

    namespace PlanetCfg
    {
        extern double g;
        extern double mu_x;
        extern double mu_y;
    } // namespace PlanetCfg

    namespace NNConfig
    {
        extern std::string name;
        extern std::vector<int> hlayers;
        extern bool use_float;
        extern int seed;
        extern int top_individuals;
        extern int population_size;
        extern bool mixed_population;
        extern bool elitism;
        extern int activation_id;
        extern bool save_nn;
        extern bool overwrite;
        extern std::string save_path_nn;
        extern int save_interval;
        extern int epochs;
        extern int layout_nb;
        extern double left_right_ratio;
        extern bool multithread;
        extern bool verbose;
    } // namespace NNConfig

    // Function to load configuration from a file
    bool loadConfiguration(const std::string& filepath);

} // namespace Config

#endif // CONFIG_LOADER_H
