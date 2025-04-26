#!/usr/bin/env python3
'''
/******************/
/* mod_lander.py  */
/*  Version 1.0   */
/*   2025/04/26   */
/******************/
'''

import pygame
import sys
import numpy as np

from mod_config import lander_cfg as lcfg


class Lander:
    def __init__(
        self, p: np.ndarray = np.zeros(2, dtype=float),
        v: np.ndarray = np.zeros(2, dtype=float),
        a: np.ndarray = np.zeros(2, dtype=float),
    ):
        self.x = p[0]
        self.y = p[1]
        self.vx = v[0]
        self.vy = v[1]
        self.ax = a[0]
        self.ay = a[1]
        self.max_fuel = float(lcfg.max_fuel)
        self.fuel = self.max_fuel
        self.width = lcfg.width
        self.height = lcfg.height
        self.vf_width = lcfg.vf_width
        self.vf_height = lcfg.vf_height
        self.vf_yoffset = lcfg.vf_yoffset
        self.hf_width = lcfg.hf_width
        self.hf_height = lcfg.hf_height
        self.hf_xoffset_l = lcfg.hf_xoffset_l
        self.hf_yoffset_l = lcfg.hf_yoffset_l
        self.hf_xoffset_r = lcfg.hf_xoffset_r
        self.hf_yoffset_r = lcfg.hf_yoffset_r
        self.image = pygame.image.load('assets/png/' + lcfg.img + '.png')
        self.vflames = pygame.image.load('assets/png/vertical_flames.png')
        self.lflames = pygame.image.load('assets/png/horizontal_flames_l.png')
        self.rflames = pygame.image.load('assets/png/horizontal_flames_r.png')

    # Position (p) and its components (x, y)


    @property
    def thrust(self):
        # give a burst of vertical speed
        self.vy += -0.3
        self.fuel -= 1

    @property
    def lthrust(self):
        self.vx -= 0.05
        self.fuel -= 0.5

    @property
    def rthrust(self):
        self.vx += 0.05
        self.fuel -= 0.5

    def debug(self):
        print(f"Position: [{self.p[0]:.2f}, {self.p[1]:.2f}], "
              f"x: {self.x:.2f}, y: {self.y:.2f}")
        print(f"Velocity: [{self.v[0]:.2f}, {self.v[1]:.2f}], "
              f"vx: {self.vx:.2f}, vy: {self.vy:.2f}")
        print(f"Acceleration: [{self.a[0]:.2f}, {self.a[1]:.2f}],"
              f" ax: {self.ax:.2f}, ay: {self.ay:.2f}")
        print(f"Fuel: {self.fuel:.2f}")
        print(f"Dimensions: [{self._dims[0]:.2f}, {self._dims[1]:.2f}], "
              f"x: {self.width:.2f}, y: {self.height:.2f}")
        print(f"Vertical Flames: [{self._vf_dims[0]:.2f}, "
              f"{self._vf_dims[1]:.2f}]")
        print(f"Horizontal Flames: [{self._hf_dims[0]:.2f}, "
              f"{self._hf_dims[1]:.2f}]")


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    pass
