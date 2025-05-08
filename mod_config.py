'''
/******************/
/*  mod_config.py */
/*  Version 2.0   */
/*   2025/05/08   */
/******************/
'''
import numpy as np
import random
from types import SimpleNamespace


RETINA_SCALE = 2

palette = SimpleNamespace(
    b=(102, 204, 255),  # blue
    o=(255, 153, 102),  # orange
    r=(204, 0, 102),    # red
    g=(102, 204, 102),  # green
    p=(204, 102, 204),  # pink
    w=(248, 248, 248),  # white
    k=(44, 44, 44)      # black
)

lander_cfg_1 = SimpleNamespace(
    width=65.5,          # width of the lander
    height=54.2,         # height of the lander
    vf_width=10,         # width of the vertical flames
    vf_height=18,        # height of the vertical flames
    vf_yoffset=20,       # offset y of the vertical flames
    hf_width=15,         # width of the horizontal flames
    hf_height=7.5,       # height of the horizontal flames
    hf_xoffset_l=-13,    # offset x of the horizontal left flames
    hf_yoffset_l=-10,    # offset y of the vertical left flames
    hf_xoffset_r=-11,    # offset x of the horizontal right flames
    hf_yoffset_r=-8,     # offset y of the vertical right flames
    max_fuel=1000,       # maximum fuel capacity for the lander
    img='lander_1'       # image name
)

lander_cfg_2 = SimpleNamespace(
    width=65.5,          # width of the lander
    height=54.2,         # height of the lander
    vf_width=10,         # width of the vertical flames
    vf_height=18,        # height of the vertical flames
    vf_yoffset=12,       # offset y of the vertical flames
    hf_width=15,         # width of the horizontal flames
    hf_height=7.5,       # height of the horizontal flames
    hf_xoffset_l=-13,    # offset x of the horizontal left flames
    hf_yoffset_l=-10,    # offset y of the vertical left flames
    hf_xoffset_r=-13,    # offset x of the horizontal right flames
    hf_yoffset_r=-10,    # offset y of the vertical right fla
    max_fuel=1000,       # maximum fuel capacity for the lander
    img='lander_2'       # image name
)

cfg = SimpleNamespace(
    width=1920 / RETINA_SCALE,     # window resolution width
    height=1080 / RETINA_SCALE,    # window resolution height
    fps=60,                        # frames per second
    save_img=False,                # save image
    save_path_img="./data/img",    # save image path
    with_sounds=False,             # use sounds
    verbose=False
)

# config used
lander_cfg = lander_cfg_1


game_cfg = SimpleNamespace(
    # --- Game Configuration ---
    # initial position [x, y] (start on takeoff pad, just above terrain)
    random_position=True,       # set randomly the pad
    x0=np.array([52.0, cfg.height - lander_cfg.height - 52]),
    v0=np.array([0.0, 0.0]),    # initial velocity [vx, vy]
    a0=np.array([0.0, 0.0]),    # initial acceleration [ax, ay]
    spad_x1=50,                 # takeoff pad left boundary (x1)
    spad_width=80,              # takeoff pad width
    lpad_x1=cfg.width - 400,    # landing pad left boundary (x1)
    lpad_width=200,             # landing pad width
    pad_y1=cfg.height - 50,     # landing/ takeoff pad top boundary (y1)
    pad_height=10,              # landing/ takeoff pad height
    terrain_y=50,               # set the zero of the terrain
    max_vx=0.5,                 # max horizontal speed when landing
    max_vy=2.0,                 # max vertical speed when landing
    max_steps=5000,             # max steps per simulation episode
    current_seed=None           # Seed used for the current pad positions
)

planet_cfg = SimpleNamespace(
    g=0.05,                     # gravitational acceleration
    mu_x=0.01,                  # friction in the x direction
    mu_y=0.01                   # friction in the y direction
)

nn_config = SimpleNamespace(
    name="lunar_lander",    # nn name
    hlayers=[16, 32, 16],   # hidden layer structure
    use_float=False,        # allow switch between c++ float and double
    seed=5247,              # seed
    top_individuals=10,     # number of top individuals to be selected
    population_size=100,    # population size
    mixed_population=True,  # use mixed population
    elitism=True,           # keep the best individual as-is
    activation_id=1,        # SIGMOID=0, TANH=1
    save_nn=True,           # save
    overwrite=False,        # save overwriting
    save_path_nn="./data/",  # save path
    save_interval=25,       # save every n generations
    epochs=1000,            # number of training epochs
    layout_nb=50,           # number of multiple layout
    left_right_ratio=0.5,   # ratio of left vs right layout
    multithread=True,       # use thread pool
    verbose=False
)


def export_cpp_config():
    # Ensure all required global variables are accessible.
    # These are: RETINA_SCALE, palette, lander_cfg, cfg, game_cfg, planet_cfg, nn_config
    # numpy (as np) should also be available for game_cfg arrays.

    # Determine which lander_cfg to use (it's globally set by 'lander_cfg = lander_cfg_1' or similar)
    active_lander_cfg = lander_cfg

    # Prepare the C++ content
    # Note: Python's True/False will be converted to C++'s true/false strings.
    # Numpy arrays like game_cfg.x0 will have their elements extracted.
    cpp_content = """\
#ifndef _CONFIG_CPP_H_303ecb31ae304863961708f4a1fa3410_
#define _CONFIG_CPP_H_303ecb31ae304863961708f4a1fa3410_

/******************/
/*  config_cpp.h  */
/*  Version 1.0   */
/*   2023/05/08   */
/******************/


#include <string>
#include <vector>
#include <array>

namespace Config {{

const int RETINA_SCALE = {retina_scale};

namespace Palette {{
    const std::array<int, 3> b = {{{b_0}, {b_1}, {b_2}}};
    const std::array<int, 3> o = {{{o_0}, {o_1}, {o_2}}};
    const std::array<int, 3> r = {{{r_0}, {r_1}, {r_2}}};
    const std::array<int, 3> g = {{{g_0}, {g_1}, {g_2}}};
    const std::array<int, 3> p = {{{p_0}, {p_1}, {p_2}}};
    const std::array<int, 3> w = {{{w_0}, {w_1}, {w_2}}};
    const std::array<int, 3> k = {{{k_0}, {k_1}, {k_2}}};
}} // namespace Palette

namespace LanderCfg {{
    const double width = {lc_width};          // width of the lander
    const double height = {lc_height};         // height of the lander
    const double vf_width = {lc_vf_width};         // width of the vertical flames
    const double vf_height = {lc_vf_height};        // height of the vertical flames
    const double vf_yoffset = {lc_vf_yoffset};       // offset y of the vertical flames
    const double hf_width = {lc_hf_width};         // width of the horizontal flames
    const double hf_height = {lc_hf_height};       // height of the horizontal flames
    const double hf_xoffset_l = {lc_hf_xoffset_l};    // offset x of the horizontal left flames
    const double hf_yoffset_l = {lc_hf_yoffset_l};    // offset y of the vertical left flames
    const double hf_xoffset_r = {lc_hf_xoffset_r};    // offset x of the horizontal right flames
    const double hf_yoffset_r = {lc_hf_yoffset_r};     // offset y of the vertical right flames
    const int max_fuel = {lc_max_fuel};       // maximum fuel capacity for the lander
    const std::string img = "{lc_img}";       // image name
}} // namespace LanderCfg

namespace Cfg {{
    const double width = {c_width};     // window resolution width
    const double height = {c_height};    // window resolution height
    const int fps = {c_fps};                        // frames per second
    const bool save_img = {c_save_img};                // save image
    const std::string save_path_img = "{c_save_path_img}";    // save image path
    const bool with_sounds = {c_with_sounds};             // use sounds
    const bool verbose = {c_verbose};
}} // namespace Cfg

namespace GameCfg {{
    const bool random_position = {gc_random_position};       // set randomly the pad
    const std::vector<double> x0 = {{{gc_x0_0}, {gc_x0_1}}};    // initial position [x, y]
    const std::vector<double> v0 = {{{gc_v0_0}, {gc_v0_1}}};    // initial velocity [vx, vy]
    const std::vector<double> a0 = {{{gc_a0_0}, {gc_a0_1}}};    // initial acceleration [ax, ay]
    const double spad_x1 = {gc_spad_x1};                 // takeoff pad left boundary (x1)
    const double spad_width = {gc_spad_width};              // takeoff pad width
    const double lpad_x1 = {gc_lpad_x1};    // landing pad left boundary (x1)
    const double lpad_width = {gc_lpad_width};             // landing pad width
    const double pad_y1 = {gc_pad_y1};     // landing/ takeoff pad top boundary (y1)
    const double pad_height = {gc_pad_height};              // landing/ takeoff pad height
    const double terrain_y = {gc_terrain_y};               // set the zero of the terrain
    const double max_vx = {gc_max_vx};                 // max horizontal speed when landing
    const double max_vy = {gc_max_vy};                 // max vertical speed when landing
    const int max_steps = {gc_max_steps};             // max steps per simulation episode
    const int current_seed = {gc_current_seed};           // Seed used for the current pad positions (None -> 0)
}} // namespace GameCfg

namespace PlanetCfg {{
    const double g = {pc_g};                     // gravitational acceleration
    const double mu_x = {pc_mu_x};                  // friction in the x direction
    const double mu_y = {pc_mu_y};                   // friction in the y direction
}} // namespace PlanetCfg

namespace NNConfig {{
    const std::string name = "{nnc_name}";    // nn name
    const std::vector<int> hlayers = {{{nnc_hlayers_str}}};   // hidden layer structure
    const bool use_float = {nnc_use_float};        // allow switch between c++ float and double
    const int seed = {nnc_seed};              // seed
    const int top_individuals = {nnc_top_individuals};     // number of top individuals to be selected
    const int population_size = {nnc_population_size};    // population size
    const bool mixed_population = {nnc_mixed_population};  // use mixed population
    const bool elitism = {nnc_elitism};           // keep the best individual as-is
    const int activation_id = {nnc_activation_id};        // SIGMOID=0, TANH=1
    const bool save_nn = {nnc_save_nn};           // save
    const bool overwrite = {nnc_overwrite};        // save overwriting
    const std::string save_path_nn = "{nnc_save_path_nn}";  // save path
    const int save_interval = {nnc_save_interval};       // save every n generations
    const int epochs = {nnc_epochs};            // number of training epochs
    const int layout_nb = {nnc_layout_nb};           // number of multiple layout
    const double left_right_ratio = {nnc_left_right_ratio};   // ratio of left vs right layout
    const bool multithread = {nnc_multihread};         // multithread
    const bool verbose = {nnc_verbose};
}}

}}

#endif // CONFIG_CPP_H
""".format(
        retina_scale=RETINA_SCALE,

        b_0=palette.b[0], b_1=palette.b[1], b_2=palette.b[2],
        o_0=palette.o[0], o_1=palette.o[1], o_2=palette.o[2],
        r_0=palette.r[0], r_1=palette.r[1], r_2=palette.r[2],
        g_0=palette.g[0], g_1=palette.g[1], g_2=palette.g[2],
        p_0=palette.p[0], p_1=palette.p[1], p_2=palette.p[2],
        w_0=palette.w[0], w_1=palette.w[1], w_2=palette.w[2],
        k_0=palette.k[0], k_1=palette.k[1], k_2=palette.k[2],

        lc_width=active_lander_cfg.width,
        lc_height=active_lander_cfg.height,
        lc_vf_width=active_lander_cfg.vf_width,
        lc_vf_height=active_lander_cfg.vf_height,
        lc_vf_yoffset=active_lander_cfg.vf_yoffset,
        lc_hf_width=active_lander_cfg.hf_width,
        lc_hf_height=active_lander_cfg.hf_height,
        lc_hf_xoffset_l=active_lander_cfg.hf_xoffset_l,
        lc_hf_yoffset_l=active_lander_cfg.hf_yoffset_l,
        lc_hf_xoffset_r=active_lander_cfg.hf_xoffset_r,
        lc_hf_yoffset_r=active_lander_cfg.hf_yoffset_r,
        lc_max_fuel=active_lander_cfg.max_fuel,
        lc_img=active_lander_cfg.img,

        c_width=cfg.width,
        c_height=cfg.height,
        c_fps=cfg.fps,
        c_save_img=str(cfg.save_img).lower(),
        c_save_path_img=cfg.save_path_img,
        c_with_sounds=str(cfg.with_sounds).lower(),
        c_verbose=str(cfg.verbose).lower(),

        gc_random_position=str(game_cfg.random_position).lower(),
        gc_x0_0=game_cfg.x0[0], gc_x0_1=game_cfg.x0[1],
        gc_v0_0=game_cfg.v0[0], gc_v0_1=game_cfg.v0[1],
        gc_a0_0=game_cfg.a0[0], gc_a0_1=game_cfg.a0[1],
        gc_spad_x1=game_cfg.spad_x1,
        gc_spad_width=game_cfg.spad_width,
        gc_lpad_x1=game_cfg.lpad_x1,
        gc_lpad_width=game_cfg.lpad_width,
        gc_pad_y1=game_cfg.pad_y1,
        gc_pad_height=game_cfg.pad_height,
        gc_terrain_y=game_cfg.terrain_y,
        gc_max_vx=game_cfg.max_vx,
        gc_max_vy=game_cfg.max_vy,
        gc_max_steps=game_cfg.max_steps,
        gc_current_seed=0 if game_cfg.current_seed is None else game_cfg.current_seed,

        pc_g=planet_cfg.g,
        pc_mu_x=planet_cfg.mu_x,
        pc_mu_y=planet_cfg.mu_y,

        nnc_name=nn_config.name,
        nnc_hlayers_str=", ".join(map(str, nn_config.hlayers)),
        nnc_use_float=str(nn_config.use_float).lower(),
        nnc_seed=nn_config.seed,
        nnc_top_individuals=nn_config.top_individuals,
        nnc_population_size=nn_config.population_size,
        nnc_mixed_population=str(nn_config.mixed_population).lower(),
        nnc_elitism=str(nn_config.elitism).lower(),
        nnc_activation_id=nn_config.activation_id,
        nnc_save_nn=str(nn_config.save_nn).lower(),
        nnc_overwrite=str(nn_config.overwrite).lower(),
        nnc_save_path_nn=nn_config.save_path_nn,
        nnc_save_interval=nn_config.save_interval,
        nnc_epochs=nn_config.epochs,
        nnc_layout_nb=nn_config.layout_nb,
        nnc_left_right_ratio=nn_config.left_right_ratio,
        nnc_multihread=str(nn_config.multithread).lower(),
        nnc_verbose=str(nn_config.verbose).lower()
    )

    file_path = "src/config_cpp.h"
    try:
        with open(file_path, "w") as f:
            f.write(cpp_content)
        print(f"Configuration successfully exported to {file_path}")
    except Exception as e:
        print(f"Error exporting configuration: {e}")


if __name__ == '__main__':
    export_cpp_config()
