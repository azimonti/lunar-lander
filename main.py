'''
/******************/
/*     main.py    */
/*  Version 2.0   */
/*   2025/04/27   */
/******************/
'''
from mod_lander import LanderVisuals
from mod_config import palette, cfg, game_cfg,  planet_cfg, lander_cfg
from mod_nn_train import NeuralNetwork
import pygame
import argparse
import glob
import numpy as np
import os
import sys
from types import SimpleNamespace

cwd = os.getcwd()
build_dir = os.path.join(cwd, "./externals/ma-libs/build")
if "DEBUG" in os.environ:
    build_path = os.path.join(build_dir, "Debug")
else:
    build_path = os.path.join(build_dir, "Release")
sys.path.append(os.path.realpath(build_path))
try:
    from lunar_lander_cpp import GameLogicCppDouble as GameLogicCpp, \
            generate_random_pad_positions
except ModuleNotFoundError as e:
    print(f"Error: {e}")

c = palette


def reset_pad_positions(seed=None, force_left_to_right=False,
                        force_right_to_left=False):
    """Resets the pad positions if random_position is True.
    Uses provided seed or generates one."""
    game_cfg.spad_x1, game_cfg.lpad_x1 = generate_random_pad_positions(
        cfg.width, game_cfg.spad_width, game_cfg.lpad_width,
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


def draw_game(screen: pygame.Surface, logic, visuals: LanderVisuals,
              sounds: SimpleNamespace, font: pygame.font.Font):
    """Renders the current game state."""
    lander_x = logic.x
    lander_y = logic.y
    fuel = logic.fuel
    vx = logic.vx
    vy = logic.vy
    last_action = logic.last_action  # Use the exposed member variable

    # Clear screen
    screen.fill(c.k)

    # Draw terrain
    pygame.draw.rect(screen, c.w, (
        0, cfg.height - game_cfg.terrain_y, cfg.width, game_cfg.terrain_y))
    # Draw pads
    pygame.draw.rect(screen, c.r, (
        game_cfg.spad_x1, game_cfg.pad_y1, game_cfg.spad_width,
        game_cfg.pad_height))
    pygame.draw.rect(screen, c.g, (
        game_cfg.lpad_x1, game_cfg.pad_y1, game_cfg.lpad_width,
        game_cfg.pad_height))

    # Draw Lander
    scaled_images = visuals.get_scaled_images()
    screen.blit(scaled_images["lander"], (lander_x, lander_y))

    # Draw Flames based on last action
    play_sound = False
    if last_action == 1 and fuel > 0:  # Up Thrust
        screen.blit(scaled_images["vflames"], (
            lander_x + visuals.width / 2 - visuals.vf_width / 2,
            lander_y + visuals.height / 2 + visuals.vf_yoffset))
        play_sound = True
    elif last_action == 2 and fuel > 0:
        screen.blit(scaled_images["rflames"], (
            lander_x + visuals.width + visuals.hf_width +
            2 * visuals.hf_xoffset_r,
            lander_y - visuals.hf_yoffset_r))
        play_sound = True
    elif last_action == 3 and fuel > 0:
        screen.blit(scaled_images["lflames"], (
            lander_x - visuals.hf_width - visuals.hf_xoffset_l,
            lander_y - visuals.hf_yoffset_l))
        play_sound = True

    # Play sound if thrusting and sound enabled
    if cfg.with_sounds and play_sound and not pygame.mixer.get_busy():
        sounds.engine_s.play()

    # Draw HUD
    fuel_text = font.render(f'Fuel: {int(fuel)}', True, c.w)
    screen.blit(fuel_text, (10, 10))
    p_text = font.render(
        f"x: {abs(lander_x):0.2f}, y: {abs(lander_y):0.2f}", True, c.w)
    screen.blit(p_text, (10, 35))
    v_text = font.render(f"vx: {abs(vx):0.2f}, vy: {abs(vy):0.2f}", True, c.w)
    screen.blit(v_text, (10, 60))

    # Update display
    pygame.display.flip()


def game_loop(mode: str):
    """Runs the main game loop for human play or NN play."""
    # Configs are now imported at the top level

    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Lunar Lander")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)

    # Initialize sound
    sounds = SimpleNamespace()
    if cfg.with_sounds:
        try:
            pygame.mixer.init()
            sounds.engine_s = pygame.mixer.Sound("assets/wav/rocket.wav")
        except pygame.error as e:
            print(f"Warning: Could not initialize sound: {e}")
            cfg.with_sounds = False  # Disable sound if init fails

    # Initialize Game Logic and Visuals
    # Use the C++ GameLogic implementation
    logic = GameLogicCpp(no_print_flag=False)
    visuals = LanderVisuals()  # Loads assets

    # --- Configure the C++ Game Logic ---
    # Configs are available from the top-level import
    logic.set_config(
        cfg_w=cfg.width,
        cfg_h=cfg.height, gcfg_pad_y1=game_cfg.pad_y1,
        gcfg_terrain_y_val=game_cfg.terrain_y,
        gcfg_max_v_x=game_cfg.max_vx, gcfg_max_v_y=game_cfg.max_vy,
        pcfg_gravity=planet_cfg.g,
        pcfg_fric_x=planet_cfg.mu_x, pcfg_fric_y=planet_cfg.mu_y,
        lcfg_w=lander_cfg.width, lcfg_h=lander_cfg.height,
        lcfg_fuel=lander_cfg.max_fuel,
        gcfg_spad_width=game_cfg.spad_width,
        gcfg_lpad_width=game_cfg.lpad_width,
        gcfg_x0_vec=game_cfg.x0.tolist(),
        gcfg_v0_vec=game_cfg.v0.tolist(),
        gcfg_a0_vec=game_cfg.a0.tolist()
    )

    # --- Reset the C++ logic using the CURRENT pad positions ---
    # This ensures the lander starts centered
    # on the potentially randomized start pad
    logic.reset(game_cfg.spad_x1, game_cfg.lpad_x1)

    # Initialize NN if in NN mode
    start_time = 0  # Initialize timer variable
    if mode == 'nn_play':
        NN = NeuralNetwork()
        NN.load()
        start_time = pygame.time.get_ticks()  # Get start time in milliseconds

    running = True
    game_over = False
    has_started = False  # Track if player has initiated movement
    frame_count = 0      # Initialize frame counter for saving images

    # Pre-allocate NumPy array for state buffer (size 5, dtype float64)
    state_buffer = np.zeros(5, dtype=np.float64)

    while running:
        clock.tick(cfg.fps)
        action = 0  # Default action: Noop

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit Key
                    running = False
                # Manual controls only active in 'play' mode
                if mode == 'play':
                    if event.key == pygame.K_UP or event.key == pygame.K_LEFT \
                            or event.key == pygame.K_RIGHT:
                        # Start game logic updates once player moves
                        has_started = True

        # --- Action Determination ---
        if not game_over:
            if mode == 'play':
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    action = 1
                elif keys[pygame.K_LEFT]:
                    action = 2
                elif keys[pygame.K_RIGHT]:
                    action = 3
            elif mode == 'nn_play':
                # Get state for NN by writing into the buffer
                logic.get_state(state_buffer)
                # Get action from NN using the buffer
                nn_action = NN.get_action(state_buffer)

                # Determine action, potentially forcing start after 2 seconds
                if not has_started:
                    elapsed_time = pygame.time.get_ticks() - start_time
                    if elapsed_time > 2000:  # 2 seconds threshold
                        print("GameLogic: Forcing start (UP) after 2 seconds.")
                        action = 1  # Force UP action
                        has_started = True
                    elif nn_action != 0:  # NN initiated action before 2s
                        action = 1  # Force UP action to start
                        has_started = True
                    else:
                        action = 0  # NOOP if NN says so and time < 2s
                else:
                    # Game already started, use NN's action
                    action = nn_action

        # --- Game Logic Update ---
        # Only update logic if game has started (player moved or NN acted)
        # and game is not over
        if has_started and not game_over:
            # logic.update now writes state into buffer and returns only done
            done = logic.update(action, state_buffer)
            if done:
                game_over = True
                # Access public members directly
                if logic.landed_successfully:
                    print("Landing Successful!")
                else:
                    print("Crashed!")
                if cfg.with_sounds and pygame.mixer.get_busy():
                    sounds.engine_s.stop()

        # --- Rendering ---
        draw_game(screen, logic, visuals, sounds, font)

        # --- Save Frame if enabled ---
        if cfg.save_img:
            filename = f"frame_{frame_count:06d}.png"
            full_path = os.path.join(cfg.save_path_img, filename)
            pygame.image.save(screen, full_path)
            frame_count += 1

        # --- Game Over Handling ---
        if game_over:
            # Display message or wait?
            pygame.time.delay(1500)  # Pause for 1.5 seconds on game over
            running = False  # End the loop after game over

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Lunar Lander Game with NN option.")
    parser.add_argument(
        '--mode',
        type=str, choices=['play', 'nn_train', 'nn_play'], default='play',
        help=("Mode to run: 'play' (manual), 'train' "
              "(NN training, no GUI), 'nn_play' (NN plays with GUI).")
    )
    parser.add_argument(
        '--continue', dest="cont", action='store_true',
        default=False, help="Continue training from checkpoint")
    parser.add_argument(
        '--step', type=int, help="Checkpoint step to load")
    parser.add_argument(
        '--left_to_right', dest="force_left_to_right", action='store_true',
        default=False, help="Force the start pad to be on the right")
    parser.add_argument(
        '--right_to_left', dest="force_right_to_left", action='store_true',
        default=False, help="Force the start pad to be on the left")
    args = parser.parse_args()

    if cfg.save_img:
        # create directory if it doesn't exist
        os.makedirs(cfg.save_path_img, exist_ok=True)
        # remove all png files in the directory
        for f in glob.glob(os.path.join(cfg.save_path_img, '*.png')):
            os.remove(f)
    elif os.path.exists(cfg.save_path_img):
        for f in glob.glob(os.path.join(cfg.save_path_img, '*.png')):
            os.remove(f)

    if args.mode == 'play':
        print("Starting game in Manual Play mode...")
        reset_pad_positions(force_left_to_right=args.force_left_to_right,
                            force_right_to_left=args.force_right_to_left)
        game_loop(mode='play')
    elif args.mode == 'nn_play':
        reset_pad_positions(force_left_to_right=args.force_left_to_right,
                            force_right_to_left=args.force_right_to_left)
        print("Starting game in NN Play mode...")
        game_loop(mode='nn_play')
    elif args.mode == 'nn_train':
        NN = NeuralNetwork()
        if args.cont:
            if args.step:
                NN.load(args.step)
            else:
                NN.load()
        else:
            NN.init()
        NN.train()

    print("Exiting.")


if __name__ == '__main__':
    main()
