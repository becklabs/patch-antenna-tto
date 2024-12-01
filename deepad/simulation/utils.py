import os
import numpy as np
import logging
from typing import List


def load_results(path: str):
    """
    Read the openEMS rectangular path simulation result
    """

    s11_arr = []
    freq_arr = []
    length_arr = []
    width_arr = []
    feed_pos_arr = []

    for file in os.listdir(path):
        if file.endswith("npz"):
            datapoint = np.load(os.path.join(path, file), allow_pickle=True)
            freqs = datapoint["frequency"]
            s11 = datapoint["s11"]
            config = datapoint["config"].item()
            length = config["length_mm"]
            width = config["width_mm"]
            feed_pos = config["feed_position_mm"]

            s11_arr.append(s11)
            freq_arr.append(freqs)
            length_arr.append(length)
            width_arr.append(width)
            feed_pos_arr.append(feed_pos)

    design_params = np.stack((length_arr, width_arr, feed_pos_arr), axis=1)
    freq_response = np.stack((freq_arr, s11_arr), axis=2)
    return design_params, freq_response


def load_preprocessed(
    data_dirs: List[str], design_params_file: str, freq_response_file: str
):
    """
    Load and stack design parameters and S11 curves from multiple data directories

    Args:
        data_dirs (List[str]): List of preprocessed sweep directories
        design_params_file (str): Design parameters file name
        freq_response_file (str): Frequency response file name

    Returns:
        tuple: (design_params, s11_curves) where each array contains stacked data
        from all directories
    """
    design_params_list = []
    s11_curves_list = []

    for data_dir in data_dirs:
        design_params_file = os.path.join(data_dir, design_params_file)
        freq_response_file = os.path.join(data_dir, freq_response_file)

        current_design_params = np.load(
            design_params_file
        )  # (num_samples, design_param_dim)
        current_freq_response = np.load(freq_response_file)  # (num_samples, N, 2)
        current_s11_curves = current_freq_response[:, :, 1]  # (num_samples, s11_length)

        design_params_list.append(current_design_params)
        s11_curves_list.append(current_s11_curves)

    design_params = np.vstack(design_params_list)  # (total_samples, design_param_dim)
    s11_curves = np.vstack(s11_curves_list)  # (total_samples, s11_length)

    return design_params, s11_curves
