import numpy as np
from typing import List


def generate_s11_curve(
    freq_range: np.array,
    resonant_freqs: List[float],
    bandwidths: List[float],
    depths_db: List[float],
) -> np.array:
    """
    Generate S11 curve with multiple Lorentzian notch resonances.

    Args:
        freq_range: np.array - Frequency points to evaluate
        resonant_freqs: List[float] - Center frequencies of resonances
        bandwidths: List[float] - Full width at half maximum for each resonance
        depths_db: List[float] - Depths of notches in dB

    Returns:
        np.array - S11 values in dB
    """
    # Initialize S11 as perfect reflection
    s11_linear = np.ones_like(freq_range, dtype=complex)

    # Add each resonance
    for f0, gamma, depth_db in zip(resonant_freqs, bandwidths, depths_db):
        delta_f = freq_range - f0

        denominator = (delta_f) ** 2 + (gamma / 2) ** 2
        notch = 1 - (gamma / 2) ** 2 / denominator  # Lorentzian notch

        depth_linear = 10 ** (depth_db / 20)
        scaled_notch = 1 - (1 - depth_linear) * (1 - notch)  # Scale to depth

        s11_linear *= scaled_notch

    s11_db = 20 * np.log10(np.abs(s11_linear))  # Convert to dB

    return s11_db
