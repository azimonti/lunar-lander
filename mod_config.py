'''
/******************/
/*  mod_config.py */
/*  Version 3.0   */
/*   2025/05/13   */
/******************/
'''
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
    x0=np.array([52.0, display_cfg.height - lander_cfg.height - 52]),
    spad_x1=50,                 # takeoff pad left boundary (x1)
    spad_width=80,              # takeoff pad width
    lpad_x1=display_cfg.width - 400,  # landing pad left boundary (x1)
    lpad_width=200,             # landing pad width
    pad_y1=display_cfg.height - 50,   # landing/ takeoff pad top boundary (y1)
    pad_height=10,              # landing/ takeoff pad height
    terrain_y=50,               # set the zero of the terrain
    current_seed=None           # Seed used for the current pad positions
)

planet_cfg = SimpleNamespace(
    g=0.05,                     # gravitational acceleration
    mu_x=0.01,                  # friction in the x direction
    mu_y=0.01                   # friction in the y direction
)

nn_config = SimpleNamespace(
    name="lunar_lander",    # neural network name
    save_path_nn="./data/"  # save path
)

if __name__ == '__main__':
    pass
