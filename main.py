#!/usr/bin/env python3
'''
/******************/
/*     main.py    */
/*  Version 1.0   */
/*   2025/04/26   */
/******************/
'''
import pygame
from types import SimpleNamespace


from mod_config import palette, cfg, game_cfg as gcfg, planet_cfg as pcfg
from mod_lander import Lander

c = palette


def loop(lander: Lander, sounds: SimpleNamespace):
    running = True
    # track if player has started moving
    has_started = False
    successful_landing = None
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    clock = pygame.time.Clock()
    lander_img = pygame.transform.scale(
        lander.image, (lander.width, lander.height))
    lander_vflames_img = pygame.transform.scale(
        lander.vflames, (lander.vf_width, lander.vf_height))
    lander_lflames_img = pygame.transform.scale(
        lander.lflames, (lander.hf_width, lander.hf_height))
    lander_rflames_img = pygame.transform.scale(
        lander.rflames, (lander.hf_width, lander.hf_height))

    while (running):
        clock.tick(cfg.fps)
        # clear screen
        screen.fill(c.k)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        # prevent the lander from going below the ground
        if lander.y >= cfg.height - lander.height - gcfg.terrain_y:
            lander.y = cfg.height - lander.height - gcfg.terrain_y
            lander.vy = 0

        screen.blit(lander_img, (lander.x, lander.y))
        # thrust
        if keys[pygame.K_UP] and lander.fuel > 0:
            lander.thrust
            screen.blit(lander_vflames_img, (
                lander.x + lander.width / 2 - lander.vf_width / 2,
                lander.y + lander.height / 2 + lander.vf_yoffset))
            has_started = True
            if cfg.with_sounds and not pygame.mixer.get_busy():
                sounds.engine_s.play()

        # lateral movement
        if keys[pygame.K_LEFT] and has_started and lander.fuel > 0:
            lander.lthrust
            screen.blit(lander_rflames_img, (
                lander.x + lander.width + lander.hf_width +
                2 * lander.hf_xoffset_r,
                lander.y - lander.hf_yoffset_r))
            if cfg.with_sounds and not pygame.mixer.get_busy():
                sounds.engine_s.play()

        if keys[pygame.K_RIGHT] and has_started and lander.fuel > 0:
            lander.rthrust
            screen.blit(lander_lflames_img, (
                lander.x - lander.hf_width - lander.hf_xoffset_l,
                lander.y - lander.hf_yoffset_l))
            if cfg.with_sounds and not pygame.mixer.get_busy():
                sounds.engine_s.play()


        # update position
        lander.x += lander.vx
        lander.y += lander.vy
        # add friction
        lander.vx *= (1 - pcfg.mu_x)
        lander.vy *= (1 - pcfg.mu_y)
        # update velocity
        lander.vy += lander.ay

        # draw terrain (ground)
        pygame.draw.rect(screen, c.w, (
            0, cfg.height - gcfg.terrain_y, cfg.width, gcfg.terrain_y))
        # draw start and end locations
        pygame.draw.rect(screen, c.r, (
            gcfg.spad_x1, gcfg.pad_y1, gcfg.spad_x2 - gcfg.spad_x1,
            gcfg.pad_y1 - gcfg.pad_y2))
        pygame.draw.rect(screen, c.g, (
            gcfg.lpad_x1, gcfg.pad_y1, gcfg.lpad_x2 - gcfg.lpad_x1,
            gcfg.pad_y1 - gcfg.pad_y2))

        # check for landing or crashing, but only after player has used fuel
        if has_started:
            if lander.y >= cfg.height - lander.height - gcfg.terrain_y:
                if lander.x >= gcfg.lpad_x1 and \
                        lander.x + lander.width <= gcfg.lpad_x2 and \
                        abs(lander.vx) < abs(gcfg.max_vx) and \
                        abs(lander.vy) < abs(gcfg.max_vy):
                    print("Successful landing!")
                    if cfg.with_sounds and pygame.mixer.get_busy():
                        sounds.engine_s.stop()
                    successful_landing = True
                else:
                    if abs(lander.vx) > abs(gcfg.max_vx):
                        print("Crashed - excessive horizontal speed!")
                    elif abs(lander.vy) > abs(gcfg.max_vy):
                        print("Crashed - excessive vertical speed!")
                    else:
                        print("Crashed - out of landing area!")
                    if cfg.with_sounds and pygame.mixer.get_busy():
                        sounds.engine_s.stop()
                    successful_landing = False
                if cfg.verbose:
                    print(f"x: {abs(lander.x):0.2f}, "
                          f"y: {abs(lander.y):0.2f}")
                    print(f"vx: {abs(lander.vx):0.2f}, "
                          f"vy: {abs(lander.vy):0.2f}")
                running = False

        font = pygame.font.Font(None, 25)
        fuel_text = font.render(f'Fuel: {int(lander.fuel)}', True, c.w)
        screen.blit(fuel_text, (10, 10))
        p_text = font.render(f"x: {abs(lander.x):0.2f}, "
                             f"y: {abs(lander.y):0.2f}",
                             True, c.w)
        screen.blit(p_text, (10, 35))
        v_text = font.render(f"vx: {abs(lander.vx):0.2f}, "
                             f"vy: {abs(lander.vy):0.2f}",
                             True, c.w)
        screen.blit(v_text, (10, 60))
        # Update display
        pygame.display.flip()
    return (has_started, successful_landing)


def main():
    a0 = gcfg.a0
    a0[1] = pcfg.g
    lander = Lander(gcfg.x0, gcfg.v0, a0)

    pygame.init()
    if cfg.with_sounds:
        # Initialize Pygame mixer
        pygame.mixer.init()
        sounds = SimpleNamespace(
            engine_s=pygame.mixer.Sound("assets/wav/rocket.wav"))
    else:
        sounds = SimpleNamespace()
    has_started, successful_landing = loop(lander, sounds)

    if has_started:
        pygame.time.delay(1000)
    pygame.quit()


if __name__ == '__main__':
    main()
