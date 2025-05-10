'''
/******************/
/*  mod_config.py */
/*  Version 2.1   */
/*   2025/05/09   */
/******************/
'''
import datetime
import numpy as np
from types import SimpleNamespace

CONFIG_VERSION = 2.1
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

display_cfg = SimpleNamespace(
    width=1920 / RETINA_SCALE,     # window resolution width
    height=1080 / RETINA_SCALE,    # window resolution height
    fps=60,                        # frames per second
    show_fitness=True,             # show neural network fitness
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
    x0=np.array([52.0, display_cfg.height - lander_cfg.height - 52]),
    v0=np.array([0.0, 0.0]),    # initial velocity [vx, vy]
    a0=np.array([0.0, 0.0]),    # initial acceleration [ax, ay]
    spad_x1=50,                 # takeoff pad left boundary (x1)
    spad_width=80,              # takeoff pad width
    lpad_x1=display_cfg.width - 400,  # landing pad left boundary (x1)
    lpad_width=200,             # landing pad width
    pad_y1=display_cfg.height - 50,   # landing/ takeoff pad top boundary (y1)
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
    name="lunar_lander",    # neural network name
    hlayers=[16, 32],       # hidden layer structure
    seed=5247,              # seed
    top_individuals=10,     # number of top individuals to be selected
    population_size=100,    # population size
    elitism=True,           # keep the best individual as-is
    activation_id=1,        # SIGMOID=0, TANH=1
    save_nn=True,           # save
    overwrite=False,        # save overwriting
    save_path_nn="./data/",  # save path
    save_interval=100,      # save every n generations
    epochs=1000,            # number of training epochs
    layout_nb=50,           # number of multiple layout
    reset_period=10,        # number of generation where 0 is the best
    left_right_ratio=0.5,   # ratio of left vs right layout
    multithread=True,       # use thread pool
    random_injection_ratio=0.60  # ratio of random individuals in population
)


def format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(map(str, value))
    elif isinstance(value, str):
        return value  # Strings are written as is
    elif value is None:
        return "0"  # Represent None as 0 for current_seed
    else:
        return str(value)


def export_text_config():
    active_lander_cfg = lander_cfg
    config_data = []

    # LanderCfg - only include width, height, max_fuel
    config_data.append("LanderCfg.width = "
                       f"{format_value(active_lander_cfg.width)}")
    config_data.append("LanderCfg.height = "
                       f"{format_value(active_lander_cfg.height)}")
    config_data.append("LanderCfg.max_fuel = "
                       f"{format_value(active_lander_cfg.max_fuel)}")

    for key, value in vars(display_cfg).items():
        config_data.append(f"DisplayCfg.{key} = {format_value(value)}")

    _game_cfg_x0 = np.array([52.0, display_cfg.height -
                             active_lander_cfg.height - 52])

    for key, value in vars(game_cfg).items():
        if key == 'x0':
            config_data.append(f"GameCfg.{key} = "
                               f"{format_value(_game_cfg_x0)}")
        elif key == 'current_seed':
            config_data.append(
                f"GameCfg.{key} = "
                f"{format_value(0 if value is None else value)}")
        else:
            config_data.append(f"GameCfg.{key} = {format_value(value)}")

    for key, value in vars(planet_cfg).items():
        config_data.append(f"PlanetCfg.{key} = {format_value(value)}")

    for key, value in vars(nn_config).items():
        config_data.append(f"NNConfig.{key} = {format_value(value)}")

    file_path = "config.txt"
    try:
        with open(file_path, "w", newline='\n') as f:
            current_date = datetime.datetime.now().strftime("%Y/%m/%d")
            f.write("# Lunar Lander Configuration File\n")
            f.write(f"# Version {CONFIG_VERSION} - "
                    f"Generated on {current_date}\n\n")
            for line in config_data:
                f.write(line + "\n")
        print(f"Configuration successfully exported to {file_path}")
    except Exception as e:
        print(f"Error exporting configuration: {e}")


if __name__ == '__main__':
    export_text_config()
