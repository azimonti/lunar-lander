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
    with_sounds=True,             # use sounds
    verbose=True
)

# config used
lander_cfg = lander_cfg_1

game_cfg = SimpleNamespace(
    # initial position [x, y] (start on takeoff pad, just above terrain)
    x0=np.array([52.0, cfg.height - lander_cfg.height - 52]),
    v0=np.array([0.0, 0.0]),    # initial velocity [vx, vy]
    a0=np.array([0.0, 0.0]),    # initial acceleration [ax, ay]
    spad_x1=50,                 # takeoff pad left boundary (x1)
    spad_x2=130,                # takeoff pad right boundary (x2)
    lpad_x1=cfg.width - 400,    # landing pad left boundary (x1)
    lpad_x2=cfg.width - 200,    # landing pad right boundary (x2)
    pad_y1=cfg.height - 50,     # landing and takeoff pad top boundary (y1)
    pad_y2=cfg.height - 60,     # landing and takeoff pad bottom boundary (y2)
    terrain_y=50,               # set the zero of the terrain
    max_vx=0.5,                 # max horizontal speed when landing
    max_vy=2.0                  # max vertical speed when landing
)

planet_cfg = SimpleNamespace(
    g=0.05,                     # gravitational acceleration
    mu_x=0.01,                  # friction in the x direction
    mu_y=0.01,                  # friction in the y direction
)


if __name__ == '__main__':
    pass
