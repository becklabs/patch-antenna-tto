import os
import numpy as np


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
