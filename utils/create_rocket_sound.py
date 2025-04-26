#!/usr/bin/env python3
'''
/**************************/
/* create_rocket_sound.py */
/*       Version 1.0      */
/*       2025/04/26       */
/**************************/
'''
import numpy as np
import sys
import wave
from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, cutoff, sample_rate, order=5):
    # low-pass filter function
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


def apply_fade_out(data, fade_duration, sample_rate):
    # fade out function
    fade_samples = int(fade_duration * sample_rate)
    fade_curve = np.linspace(1, 0, fade_samples)
    data[-fade_samples:] = (data[-fade_samples:] * fade_curve).astype(np.int16)
    return data


def main():
    # parameters
    duration = 0.5  # 500 ms
    fade_duration = 0.1  # 100 ms fade out
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    amplitude = 32767
    cutoff_freq = 500  # low-pass filter cutoff

    # generate white noise
    noise = np.random.normal(0, amplitude, n_samples).astype(np.int16)

    # apply low-pass filter to lower the pitch
    filtered_noise = butter_lowpass_filter(
        noise, cutoff_freq, sample_rate).astype(np.int16)

    # apply fade-out to the last 100 ms
    filtered_noise = apply_fade_out(
        filtered_noise, fade_duration, sample_rate)

    with wave.open("assets/wav/rocket.wav", "w") as wav_file:
        wav_file.setparams((
            1, 2, sample_rate, n_samples, "NONE", "not compressed"))
        wav_file.writeframes(filtered_noise.tobytes())

    print("assets/wav/rocket.wav saved.")


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    main()
