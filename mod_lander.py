'''
/******************/
/* mod_lander.py  */
/*  Version 1.0   */
/*   2025/04/26   */
/******************/
'''

import pygame

from mod_config import lander_cfg


class LanderVisuals:
    """Holds the visual assets and static properties of the lander."""

    def __init__(self):
        # Static properties from config
        self.width = lander_cfg.width
        self.height = lander_cfg.height
        self.vf_width = lander_cfg.vf_width
        self.vf_height = lander_cfg.vf_height
        self.vf_yoffset = lander_cfg.vf_yoffset
        self.hf_width = lander_cfg.hf_width
        self.hf_height = lander_cfg.hf_height
        self.hf_xoffset_l = lander_cfg.hf_xoffset_l
        self.hf_yoffset_l = lander_cfg.hf_yoffset_l
        self.hf_xoffset_r = lander_cfg.hf_xoffset_r
        self.hf_yoffset_r = lander_cfg.hf_yoffset_r

        # Load assets
        try:
            self.image = pygame.image.load(
                'assets/png/' + lander_cfg.img + '.png').convert_alpha()
            self.vflames = pygame.image.load(
                'assets/png/vertical_flames.png').convert_alpha()
            self.lflames = pygame.image.load(
                'assets/png/horizontal_flames_l.png').convert_alpha()
            self.rflames = pygame.image.load(
                'assets/png/horizontal_flames_r.png').convert_alpha()
        except pygame.error as e:
            print(f"Error loading lander assets: {e}")

        # Pre-scale images (optional, can also be done in the draw loop)
        self.lander_img_scaled = pygame.transform.scale(
            self.image, (self.width, self.height))
        self.vflames_img_scaled = pygame.transform.scale(
            self.vflames, (self.vf_width, self.vf_height))
        self.lflames_img_scaled = pygame.transform.scale(
            self.lflames, (self.hf_width, self.hf_height))
        self.rflames_img_scaled = pygame.transform.scale(
            self.rflames, (self.hf_width, self.hf_height))

    def get_scaled_images(self):
        """Returns the pre-scaled images."""
        return {
            "lander": self.lander_img_scaled,
            "vflames": self.vflames_img_scaled,
            "lflames": self.lflames_img_scaled,
            "rflames": self.rflames_img_scaled
        }


# Example usage or test (optional)
if __name__ == '__main__':
    # Initialize pygame minimally to load images if run directly
    pygame.init()
    pygame.display.set_mode((1, 1))  # Dummy display needed for image loading

    try:
        lander_visuals = LanderVisuals()
        print("LanderVisuals initialized successfully.")
        print(
            f" Lander dimensions: {lander_visuals.width}x"
            f"{lander_visuals.height}")
        images = lander_visuals.get_scaled_images()
        print(f" Loaded {len(images)} scaled images.")
    except Exception as e:
        print(f"Error initializing LanderVisuals: {e}")
    finally:
        pygame.quit()
