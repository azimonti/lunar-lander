#!/usr/bin/env python3
'''
/******************/
/*     main.py    */
/*  Version 2.0   */
/*   2025/04/27   */
/******************/
'''
import pygame
import argparse
import sys
import numpy as np
from types import SimpleNamespace

from mod_config import palette, cfg, game_cfg as gcfg, planet_cfg as pcfg, lander_cfg as lcfg
from game_logic import GameLogic
from mod_lander import LanderVisuals # Renamed from Lander
# Import NN functions and training loop from train_nn.py
from train_nn import initialize_nn, get_nn_action, run_training_loop

c = palette


def draw_game(screen: pygame.Surface, logic: GameLogic, visuals: LanderVisuals, sounds: SimpleNamespace, font: pygame.font.Font):
    """Renders the current game state."""
    render_info = logic.get_render_info()
    lander_x = render_info["x"]
    lander_y = render_info["y"]
    fuel = render_info["fuel"]
    vx = render_info["vx"]
    vy = render_info["vy"]
    last_action = render_info["last_action"]

    # Clear screen
    screen.fill(c.k)

    # Draw terrain
    pygame.draw.rect(screen, c.w, (
        0, cfg.height - gcfg.terrain_y, cfg.width, gcfg.terrain_y))
    # Draw pads
    pygame.draw.rect(screen, c.r, (
        gcfg.spad_x1, gcfg.pad_y1, gcfg.spad_x2 - gcfg.spad_x1,
        gcfg.pad_y1 - gcfg.pad_y2))
    pygame.draw.rect(screen, c.g, (
        gcfg.lpad_x1, gcfg.pad_y1, gcfg.lpad_x2 - gcfg.lpad_x1,
        gcfg.pad_y1 - gcfg.pad_y2))

    # Draw Lander
    scaled_images = visuals.get_scaled_images()
    screen.blit(scaled_images["lander"], (lander_x, lander_y))

    # Draw Flames based on last action
    play_sound = False
    if last_action == 1 and fuel > 0: # Up Thrust
        screen.blit(scaled_images["vflames"], (
            lander_x + visuals.width / 2 - visuals.vf_width / 2,
            lander_y + visuals.height / 2 + visuals.vf_yoffset))
        play_sound = True
    elif last_action == 2 and fuel > 0: # Left Thrust
        screen.blit(scaled_images["rflames"], ( # Right flames for left thrust
            lander_x + visuals.width + visuals.hf_width +
            2 * visuals.hf_xoffset_r,
            lander_y - visuals.hf_yoffset_r))
        play_sound = True
    elif last_action == 3 and fuel > 0: # Right Thrust
        screen.blit(scaled_images["lflames"], ( # Left flames for right thrust
            lander_x - visuals.hf_width - visuals.hf_xoffset_l,
            lander_y - visuals.hf_yoffset_l))
        play_sound = True

    # Play sound if thrusting and sound enabled
    if cfg.with_sounds and play_sound and not pygame.mixer.get_busy():
        sounds.engine_s.play()

    # Draw HUD
    fuel_text = font.render(f'Fuel: {int(fuel)}', True, c.w)
    screen.blit(fuel_text, (10, 10))
    p_text = font.render(f"x: {abs(lander_x):0.2f}, y: {abs(lander_y):0.2f}", True, c.w)
    screen.blit(p_text, (10, 35))
    v_text = font.render(f"vx: {abs(vx):0.2f}, vy: {abs(vy):0.2f}", True, c.w)
    screen.blit(v_text, (10, 60))

    # Update display
    pygame.display.flip()


def game_loop(mode: str):
    """Runs the main game loop for human play or NN play."""
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
            cfg.with_sounds = False # Disable sound if init fails

    # Initialize Game Logic and Visuals
    logic = GameLogic()
    visuals = LanderVisuals() # Loads assets

    # Initialize NN if in NN mode
    if mode == 'nn_play':
        initialize_nn()

    running = True
    game_over = False
    has_started = False # Track if player has initiated movement

    while running:
        clock.tick(cfg.fps)
        action = 0 # Default action: Noop

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: # Quit Key
                    running = False
                # Manual controls only active in 'play' mode
                if mode == 'play':
                    if event.key == pygame.K_UP or event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                         has_started = True # Start game logic updates once player moves

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
                 # Get state for NN
                 current_state = logic.get_state()
                 # Get action from NN
                 action = get_nn_action(current_state)
                 # NN always "starts" the game
                 if not has_started and action != 0:
                     has_started = True


        # --- Game Logic Update ---
        # Only update logic if game has started (player moved or NN acted)
        # and game is not over
        if has_started and not game_over:
            state, reward, done = logic.update(action)
            if done:
                game_over = True
                render_info = logic.get_render_info()
                if render_info["landed_successfully"]:
                    print("Landing Successful!")
                else:
                    print("Crashed!")
                if cfg.with_sounds and pygame.mixer.get_busy():
                    sounds.engine_s.stop()


        # --- Rendering ---
        draw_game(screen, logic, visuals, sounds, font)

        # --- Game Over Handling ---
        if game_over:
            # Display message or wait?
            pygame.time.delay(1500) # Pause for 1.5 seconds on game over
            running = False # End the loop after game over


    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Lunar Lander Game with NN option.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['play', 'train', 'nn_play'],
        default='play',
        help="Mode to run: 'play' (manual), 'train' (NN training, no GUI), 'nn_play' (NN plays with GUI)."
    )
    args = parser.parse_args()

    if args.mode == 'play':
        print("Starting game in Manual Play mode...")
        game_loop(mode='play')
    elif args.mode == 'nn_play':
        print("Starting game in NN Play mode...")
        game_loop(mode='nn_play')
    elif args.mode == 'train':
        run_training_loop() # Call the training function

    print("Exiting.")


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    main()
