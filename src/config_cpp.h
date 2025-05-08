#ifndef _CONFIG_CPP_H_303ecb31ae304863961708f4a1fa3410_
#define _CONFIG_CPP_H_303ecb31ae304863961708f4a1fa3410_

/******************/
/*  config_cpp.h  */
/*  Version 1.0   */
/*   2023/05/08   */
/******************/

#include <array>
#include <string>
#include <vector>

namespace Config
{

    const int RETINA_SCALE = 2;

    namespace Palette
    {
        const std::array<int, 3> b = {102, 204, 255};
        const std::array<int, 3> o = {255, 153, 102};
        const std::array<int, 3> r = {204, 0, 102};
        const std::array<int, 3> g = {102, 204, 102};
        const std::array<int, 3> p = {204, 102, 204};
        const std::array<int, 3> w = {248, 248, 248};
        const std::array<int, 3> k = {44, 44, 44};
    } // namespace Palette

    namespace LanderCfg
    {
        const double width        = 65.5;       // width of the lander
        const double height       = 54.2;       // height of the lander
        const double vf_width     = 10;         // width of the vertical flames
        const double vf_height    = 18;         // height of the vertical flames
        const double vf_yoffset   = 20;         // offset y of the vertical flames
        const double hf_width     = 15;         // width of the horizontal flames
        const double hf_height    = 7.5;        // height of the horizontal flames
        const double hf_xoffset_l = -13;        // offset x of the horizontal left flames
        const double hf_yoffset_l = -10;        // offset y of the vertical left flames
        const double hf_xoffset_r = -11;        // offset x of the horizontal right flames
        const double hf_yoffset_r = -8;         // offset y of the vertical right flames
        const int max_fuel        = 1000;       // maximum fuel capacity for the lander
        const std::string img     = "lander_1"; // image name
    } // namespace LanderCfg

    namespace Cfg
    {
        const double width              = 960.0;        // window resolution width
        const double height             = 540.0;        // window resolution height
        const int fps                   = 60;           // frames per second
        const bool save_img             = false;        // save image
        const std::string save_path_img = "./data/img"; // save image path
        const bool with_sounds          = false;        // use sounds
        const bool verbose              = false;
    } // namespace Cfg

    namespace GameCfg
    {
        const bool random_position   = true;          // set randomly the pad
        const std::vector<double> x0 = {52.0, 433.8}; // initial position [x, y]
        const std::vector<double> v0 = {0.0, 0.0};    // initial velocity [vx, vy]
        const std::vector<double> a0 = {0.0, 0.0};    // initial acceleration [ax, ay]
        const double spad_x1         = 50;            // takeoff pad left boundary (x1)
        const double spad_width      = 80;            // takeoff pad width
        const double lpad_x1         = 560.0;         // landing pad left boundary (x1)
        const double lpad_width      = 200;           // landing pad width
        const double pad_y1          = 490.0;         // landing/ takeoff pad top boundary (y1)
        const double pad_height      = 10;            // landing/ takeoff pad height
        const double terrain_y       = 50;            // set the zero of the terrain
        const double max_vx          = 0.5;           // max horizontal speed when landing
        const double max_vy          = 2.0;           // max vertical speed when landing
        const int max_steps          = 5000;          // max steps per simulation episode
        const int current_seed       = 0;             // Seed used for the current pad positions (None -> 0)
    } // namespace GameCfg

    namespace PlanetCfg
    {
        const double g    = 0.05; // gravitational acceleration
        const double mu_x = 0.01; // friction in the x direction
        const double mu_y = 0.01; // friction in the y direction
    } // namespace PlanetCfg

    namespace NNConfig
    {
        const std::string name         = "lunar_lander"; // nn name
        const std::vector<int> hlayers = {16, 32, 16};   // hidden layer structure
        const bool use_float           = false;          // allow switch between c++ float and double
        const int seed                 = 5247;           // seed
        const int top_individuals      = 10;             // number of top individuals to be selected
        const int population_size      = 100;            // population size
        const bool mixed_population    = true;           // use mixed population
        const bool elitism             = true;           // keep the best individual as-is
        const int activation_id        = 1;              // SIGMOID=0, TANH=1
        const bool save_nn             = true;           // save
        const bool overwrite           = false;          // save overwriting
        const std::string save_path_nn = "./data/";      // save path
        const int save_interval        = 25;             // save every n generations
        const int epochs               = 1000;           // number of training epochs
        const int layout_nb            = 50;             // number of multiple layout
        const double left_right_ratio  = 0.5;            // ratio of left vs right layout
        const bool verbose             = false;
    } // namespace NNConfig

} // namespace Config

#endif // CONFIG_CPP_H
