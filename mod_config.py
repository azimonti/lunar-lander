'''
/******************/
/*  mod_config.py */
/*  Version 1.0   */
/*   2024/09/10   */
/******************/
'''
import numpy as np
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
        save_path="./data/img",        # save image path
        with_sounds=False,             # use sounds
        verbose=True
        )

# config used
lander_cfg = lander_cfg_1

game_cfg = SimpleNamespace(
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
        pad_height = 10,            # landing/ takeoff pad height
        terrain_y=50,               # set the zero of the terrain
        max_vx=0.5,                 # max horizontal speed when landing
        max_vy=2.0,                 # max vertical speed when landing
        max_steps=1000              # max steps per simulation episode
        )

planet_cfg = SimpleNamespace(
        g=0.05,                     # gravitational acceleration
        mu_x=0.01,                  # friction in the x direction
        mu_y=0.01                   # friction in the y direction
        )

nn_config = SimpleNamespace(
    name="lunar_lander",    # nn name
    hlayers=[8, 16, 8],     # hidden layer structure
    seed=5247,              # seed
    top_individuals=10,     # number of top individuals to be selected
    population_size=300,    # population size
    mixed_population=True,  # use mixed population
    elitism=True,           # keep the best individual as-is
    activation_id=1,        # SIGMOID=0, TANH=1
    epochsThread=100,       # number of epochs to Train in a single run
    save_nn=True,           # save
    overwrite=False,        # save overwriting
    save_path_nn="./data/", # save path
    save_interval=25,       # save every n generations
    epochs=1000,            # number of training epochs
    verbose=False
    )

if __name__ == '__main__':
    pass
