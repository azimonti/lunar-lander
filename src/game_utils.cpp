/********************/
/*  game_utils.cpp  */
/*   Version 1.0    */
/*    2023/05/08    */
/********************/
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include "game_utils.h"

std::pair<int, int> generate_random_pad_positions(double screen_width_double, int spad_width, int lpad_width,
                                                  std::optional<int> seed_opt, bool force_left_to_right,
                                                  bool force_right_to_left)
{

    const int MIN_PAD_SEPARATION = 10;
    const int BORDER_LEFT        = 10;
    const int BORDER_RIGHT       = 10;
    const int screen_width       = static_cast<int>(screen_width_double);

    std::mt19937 rng;
    if (seed_opt.has_value()) { rng.seed(static_cast<unsigned int>(seed_opt.value())); }
    else
    {
        std::random_device rd;
        rng.seed(rd());
    }

    // Distribution for pad starting positions.
    std::uniform_int_distribution<int> dist(BORDER_LEFT, screen_width);

    int spad_x1 = 0;
    int lpad_x1 = 0;

    while (true)
    {
        spad_x1      = dist(rng);
        lpad_x1      = dist(rng);

        // 1. Check Right Boundary
        bool spad_ok = (spad_x1 + spad_width <= (screen_width - BORDER_RIGHT));
        bool lpad_ok = (lpad_x1 + lpad_width <= (screen_width - BORDER_RIGHT));

        if (!spad_ok || !lpad_ok) { continue; }

        // 2. Check Separation
        double center_spad = static_cast<double>(spad_x1) + static_cast<double>(spad_width) / 2.0;
        double center_lpad = static_cast<double>(lpad_x1) + static_cast<double>(lpad_width) / 2.0;
        double min_dist_centers =
            (static_cast<double>(spad_width) / 2.0) + (static_cast<double>(lpad_width) / 2.0) + MIN_PAD_SEPARATION;

        if (std::abs(center_spad - center_lpad) >= min_dist_centers)
        {
            bool needs_mirror = false;
            if (force_left_to_right && (center_spad >= center_lpad)) { needs_mirror = true; }
            else if (force_right_to_left && (center_spad <= center_lpad)) { needs_mirror = true; }

            if (needs_mirror)
            {
                double mid_point = static_cast<double>(BORDER_LEFT) +
                                   (static_cast<double>(screen_width - BORDER_LEFT - BORDER_RIGHT)) / 2.0;

                double mirrored_center_spad = 2.0 * mid_point - center_spad;
                double mirrored_center_lpad = 2.0 * mid_point - center_lpad;

                spad_x1 = static_cast<int>(mirrored_center_spad - static_cast<double>(spad_width) / 2.0);
                lpad_x1 = static_cast<int>(mirrored_center_lpad - static_cast<double>(lpad_width) / 2.0);
            }
            break; // Valid configuration found
        }
    }
    return std::make_pair(spad_x1, lpad_x1);
}
