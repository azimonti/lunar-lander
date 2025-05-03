'''
/******************/
/*  mod_config.py */
/*  Version 1.0   */
/*   2024/09/10   */
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
    max_steps=1000,             # max steps per simulation episode
    current_seed=None           # Seed used for the current pad positions
)


def generate_random_pad_positions(seed=None,
                                  force_left_to_right=False,
                                  force_right_to_left=False):
    """Generates random, non-overlapping positions
       for start and landing pads
       using a simple generate-and-test approach."""
    MIN_PAD_SEPARATION = 10
    BORDER_LEFT = 10
    BORDER_RIGHT = 10
    screen_width = cfg.width
    spad_width = game_cfg.spad_width
    lpad_width = game_cfg.lpad_width
    if seed is not None:
        random.seed(seed)

    # Loop indefinitely until a valid configuration is found
    while True:
        # Generate random starting points anywhere on the screen width
        # Note: randint includes the upper bound, so screen_width is possible,
        # but the right edge check below will handle it.
        spad_x1 = random.randint(BORDER_LEFT, int(screen_width))
        lpad_x1 = random.randint(BORDER_LEFT, int(screen_width))

        # 1. Check Right Boundary: Ensure pads don't go off the right edge
        spad_ok = (spad_x1 + spad_width <= (screen_width - BORDER_RIGHT))
        lpad_ok = (lpad_x1 + lpad_width <= (screen_width - BORDER_RIGHT))

        if not spad_ok or not lpad_ok:
            continue  # Invalid placement, try again

        # 2. Check Separation: Ensure pads don't overlap
        # and have minimum separation
        center_spad = spad_x1 + spad_width / 2
        center_lpad = lpad_x1 + lpad_width / 2
        min_dist_centers = (spad_width / 2) + \
            (lpad_width / 2) + MIN_PAD_SEPARATION

        if abs(center_spad - center_lpad) >= min_dist_centers:
            # Valid geometric configuration found
            needs_mirror = False
            if force_left_to_right and (center_spad >= center_lpad):
                # Spad is right/equal, but needs to be left -> mirror
                needs_mirror = True
            elif force_right_to_left and (center_spad <= center_lpad):
                # Spad is left/equal, but needs to be right -> mirror
                needs_mirror = True

            if needs_mirror:
                # Mirror positions relative to the center of the usable area
                mid_point = BORDER_LEFT + (
                    screen_width - BORDER_LEFT - BORDER_RIGHT) / 2.0

                # Mirror centers
                mirrored_center_spad = 2 * mid_point - center_spad
                mirrored_center_lpad = 2 * mid_point - center_lpad

                # Update x1 positions based on mirrored centers
                spad_x1 = mirrored_center_spad - spad_width / 2.0
                lpad_x1 = mirrored_center_lpad - lpad_width / 2.0

            # Configuration is now geometrically valid
            # and directionally correct (if forced)
            break  # Exit the while loop

        # else: continue loop to generate a new pair

    # Return the validated pair
    return int(spad_x1), int(lpad_x1)


def reset_pad_positions(seed=None, force_left_to_right=False,
                        force_right_to_left=False):
    """Resets the pad positions if random_position is True.
    Uses provided seed or generates one."""
    # global game_cfg  # Ensure we modify the global game_cfg
    if game_cfg.random_position:
        if seed is None:
            # Generate a new seed if none provided for reproducibility tracking
            seed = random.randint(0, 2**32 - 1)
        game_cfg.current_seed = seed
        game_cfg.spad_x1, game_cfg.lpad_x1 = generate_random_pad_positions(
            seed=game_cfg.current_seed,
            force_left_to_right=force_left_to_right,
            force_right_to_left=force_right_to_left
        )
        # Update lander initial x position to be centered on the new start pad
        game_cfg.x0[0] = game_cfg.spad_x1 + (game_cfg.spad_width / 2
                                             ) - (lander_cfg.width / 2)

        if cfg.verbose:
            print(f"Pad positions random with seed: {game_cfg.current_seed}")
            print(f"  Start Pad x1: {game_cfg.spad_x1}, "
                  f"Landing Pad x1: {game_cfg.lpad_x1}")
            print(f"  Lander Initial x0: {game_cfg.x0[0]:.2f}")


def set_pad_positions(spad_x1, lpad_x1):
    """Sets the pad positions to specific values."""
    # global game_cfg, lander_cfg, cfg # Ensure access if needed

    game_cfg.spad_x1 = spad_x1
    game_cfg.lpad_x1 = lpad_x1

    # Update lander initial x position based on the new start pad
    game_cfg.x0[0] = game_cfg.spad_x1 + (game_cfg.spad_width / 2
                                         ) - (lander_cfg.width / 2)

    # Optionally log the manual setting
    if cfg.verbose:
        print("Pad positions set manually:")
        print(f"  Start Pad x1: {game_cfg.spad_x1}, "
              f"Landing Pad x1: {game_cfg.lpad_x1}")
        print(f"  Lander Initial x0: {game_cfg.x0[0]:.2f}")


planet_cfg = SimpleNamespace(
    g=0.05,                     # gravitational acceleration
    mu_x=0.01,                  # friction in the x direction
    mu_y=0.01                   # friction in the y direction
)

nn_config = SimpleNamespace(
    name="lunar_lander",    # nn name
    hlayers=[8, 64, 16],    # hidden layer structure
    seed=5247,              # seed
    top_individuals=10,     # number of top individuals to be selected
    population_size=300,    # population size
    mixed_population=True,  # use mixed population
    elitism=True,           # keep the best individual as-is
    activation_id=1,        # SIGMOID=0, TANH=1
    save_nn=True,           # save
    overwrite=False,        # save overwriting
    save_path_nn="./data/",  # save path
    save_interval=25,       # save every n generations
    epochs=1000,            # number of training epochs
    nb_batches=100,         # if random_position=True keep fix the layout
    fit_min=-2000,          # reset if going below for more than 5 generations
    fit_streak=5,           # max consecutive low fitness generations
    verbose=False
)

if __name__ == '__main__':
    pass
