'''
/******************/
/*     main.py    */
/*  Version 2.1   */
/*   2025/05/10   */
/******************/
'''
from mod_lander import LanderVisuals
from mod_config import palette, game_cfg, lander_cfg, display_cfg
from mod_nn_play import NeuralNetwork
import pygame
import argparse
import glob
import numpy as np
import os
import sys
from types import SimpleNamespace

cwd = os.getcwd()
build_dir = os.path.join(cwd, "./externals/ma-libs/build/Release")
sys.path.append(os.path.realpath(build_dir))
try:
    from lunar_lander_cpp import GameLogicCppDouble as GameLogicCpp, \
        generate_random_pad_positions
except ModuleNotFoundError as e:
    print(f"Error: {e}")

c = palette


def draw_game(screen: pygame.Surface, logic, visuals: LanderVisuals,
              sounds: SimpleNamespace, font: pygame.font.Font,
              mode: str = 'play', current_fitness_to_display: float = 0.0):
    """Renders the current game state."""
    lander_x = logic.x
    lander_y = logic.y
    fuel = logic.fuel
    vx = logic.vx
    vy = logic.vy
    last_action = logic.last_action

    # Clear screen
    screen.fill(c.k)

    # Draw terrain
    pygame.draw.rect(screen, c.w, (
        0, display_cfg.height - game_cfg.terrain_y, display_cfg.width,
        game_cfg.terrain_y))
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
    if display_cfg.with_sounds and play_sound and not pygame.mixer.get_busy():
        sounds.engine_s.play()

    # Draw HUD
    fuel_text = font.render(f'Fuel: {int(fuel)}', True, c.w)
    screen.blit(fuel_text, (10, 10))
    p_text = font.render(
        f"x: {abs(lander_x):0.2f}, y: {abs(lander_y):0.2f}", True, c.w)
    screen.blit(p_text, (10, 35))
    v_text = font.render(f"vx: {abs(vx):0.2f}, vy: {abs(vy):0.2f}", True, c.w)
    screen.blit(v_text, (10, 60))

    # Display fitness if in NN mode and enabled
    if mode == 'nn_play' and display_cfg.show_fitness:
        fitness_text_surf = font.render("Fitness: "
                                        f"{current_fitness_to_display:.2f}",
                                        True, c.w)
        screen.blit(fitness_text_surf, (10, 85))

    # Update display
    pygame.display.flip()


def game_loop(mode: str, force_left_to_right: bool = False,
              force_right_to_left: bool = False):
    """Runs the main game loop for human play or NN play."""
    # Configs are now imported at the top level

    pygame.init()
    screen = pygame.display.set_mode((display_cfg.width, display_cfg.height))
    pygame.display.set_caption("Lunar Lander")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)

    # Initialize sound
    sounds = SimpleNamespace()
    if display_cfg.with_sounds:
        try:
            pygame.mixer.init()
            sounds.engine_s = pygame.mixer.Sound("assets/wav/rocket.wav")
        except pygame.error as e:
            print(f"Warning: Could not initialize sound: {e}")
            display_cfg.with_sounds = False  # Disable sound if init fails

    # Initialize Game Logic and Visuals
    # Use the C++ GameLogic implementation
    logic = GameLogicCpp(False, 'config.txt')
    visuals = LanderVisuals()  # Loads assets

    # Reset pad positions
    game_cfg.spad_x1, game_cfg.lpad_x1 = generate_random_pad_positions(
        display_cfg.width, game_cfg.spad_width, game_cfg.lpad_width,
        seed=game_cfg.current_seed,
        force_left_to_right=force_left_to_right,
        force_right_to_left=force_right_to_left
    )
    # Update lander initial x position to be centered on the new start pad
    game_cfg.x0[0] = game_cfg.spad_x1 + (game_cfg.spad_width / 2
                                         ) - (lander_cfg.width / 2)

    # Update game logic with new pad positions
    logic.reset(game_cfg.spad_x1, game_cfg.lpad_x1)

    # Initialize NN if in NN mode
    start_time = 0  # Initialize timer variable
    if mode == 'nn_play':
        NN = NeuralNetwork()
        start_time = pygame.time.get_ticks()  # Get start time in milliseconds

    running = True
    game_over = False
    has_started = False  # Track if player has initiated movement
    frame_count = 0      # Initialize frame counter for saving images

    # Fitness tracking for NN mode
    penalty = 0.0
    current_game_steps = 0

    # Pre-allocate NumPy array for state buffer (size 5, dtype float64)
    state_buffer = np.zeros(5, dtype=np.float64)

    while running:
        clock.tick(display_cfg.fps)
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

            if mode == 'nn_play':
                step_penalty = logic.calculate_step_penalty(action)
                penalty += step_penalty
                current_game_steps += 1

            if done:
                game_over = True
                # Access public members directly
                if logic.landed_successfully:
                    print("Landing Successful!")
                else:
                    print("Crashed!")
                if display_cfg.with_sounds and pygame.mixer.get_busy():
                    sounds.engine_s.stop()

        if mode == 'nn_play':
            if game_over:
                terminal_penalty = logic.calculate_terminal_penalty(
                    current_game_steps)
                penalty += terminal_penalty

        # --- Rendering ---
        if mode == 'nn_play':
            draw_game(screen, logic, visuals, sounds, font,
                      mode=mode, current_fitness_to_display=penalty)
        else:
            draw_game(screen, logic, visuals, sounds, font)

        # --- Save Frame if enabled ---
        if display_cfg.save_img:
            filename = f"frame_{frame_count:06d}.png"
            full_path = os.path.join(display_cfg.save_path_img, filename)
            pygame.image.save(screen, full_path)
            frame_count += 1

        # --- Game Over Handling ---
        if game_over:
            message_text = ""
            message_color = c.w
            if logic.landed_successfully:
                message_text = "LANDED!"
                message_color = c.g
            else:
                message_text = "CRASHED!"
                message_color = c.r

            # Render the message
            msg_font = pygame.font.Font(None, 74)  # Larger font for game over
            text_surface = msg_font.render(message_text, True, message_color)
            text_rect = text_surface.get_rect(
                center=(display_cfg.width // 2, display_cfg.height // 2))

            # Blit message on top of the last drawn frame
            screen.blit(text_surface, text_rect)

            # Loop for 2 seconds to display message and save frames
            game_over_display_duration_ms = 2000
            game_over_loop_start_time = pygame.time.get_ticks()

            while (pygame.time.get_ticks() - game_over_loop_start_time
                   < game_over_display_duration_ms) and running:
                # Handle events to allow quitting
                # during the game over message display
                for event_game_over in pygame.event.get():
                    if event_game_over.type == pygame.QUIT:
                        running = False
                        break
                    if event_game_over.type == pygame.KEYDOWN:
                        if event_game_over.key == pygame.K_q:
                            running = False
                            break

                if not running:
                    break

                pygame.display.flip()

                # Save frame if enabled
                if display_cfg.save_img:
                    filename = f"frame_{frame_count:06d}.png"
                    full_path = os.path.join(
                            display_cfg.save_path_img, filename)
                    pygame.image.save(screen, full_path)
                    frame_count += 1

                clock.tick(display_cfg.fps)  # Maintain FPS

            running = False

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Lunar Lander Game with NN option.")
    parser.add_argument(
        '--mode',
        type=str, choices=['play', 'nn_play'], default='play',
        help=("Mode to run: 'play' (manual), 'train' "
              "(NN training, no GUI), 'nn_play' (NN plays with GUI).")
    )
    parser.add_argument(
        '--left_to_right', dest="force_left_to_right", action='store_true',
        default=False, help="Force the start pad to be on the right")
    parser.add_argument(
        '--right_to_left', dest="force_right_to_left", action='store_true',
        default=False, help="Force the start pad to be on the left")
    args = parser.parse_args()

    if display_cfg.save_img:
        # create directory if it doesn't exist
        os.makedirs(display_cfg.save_path_img, exist_ok=True)
        # remove all png files in the directory
        for f in glob.glob(os.path.join(display_cfg.save_path_img, '*.png')):
            os.remove(f)
    elif os.path.exists(display_cfg.save_path_img):
        for f in glob.glob(os.path.join(display_cfg.save_path_img, '*.png')):
            os.remove(f)

    if args.mode == 'play':
        print("Starting game in Manual Play mode...")
        game_loop(mode='play',
                  force_left_to_right=args.force_left_to_right,
                  force_right_to_left=args.force_right_to_left)
    elif args.mode == 'nn_play':
        print("Starting game in NN Play mode...")
        game_loop(mode='nn_play',
                  force_left_to_right=args.force_left_to_right,
                  force_right_to_left=args.force_right_to_left)

    print("Exiting.")


if __name__ == '__main__':
    main()
