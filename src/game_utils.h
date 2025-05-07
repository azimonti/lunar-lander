#ifndef _GAME_UTILS_H_DB85736B89AB440CA14107A22F8C1DD5_
#define _GAME_UTILS_H_DB85736B89AB440CA14107A22F8C1DD5_

/******************/
/*  game_utils.h  */
/*  Version 1.0   */
/*   2023/05/08   */
/******************/

#include <optional>
#include <utility>
#include <vector>

// Function to generate random, non-overlapping positions for start and landing pads.
// screen_width: The total width of the game screen.
// spad_width: The width of the starting pad.
// lpad_width: The width of the landing pad.
// seed: Optional seed for the random number generator.
// force_left_to_right: If true, ensures the starting pad is to the left of the landing pad.
// force_right_to_left: If true, ensures the starting pad is to the right of the landing pad.
// Returns a pair of integers: (spad_x1, lpad_x1)
std::pair<int, int> generate_random_pad_positions(double screen_width, int spad_width, int lpad_width,
                                                  std::optional<int> seed  = std::nullopt,
                                                  bool force_left_to_right = false, bool force_right_to_left = false);

#endif
