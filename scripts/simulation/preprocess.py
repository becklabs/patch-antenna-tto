import os
import numpy as np
import argparse
from patchtto.simulation.utils import load_results

RESONANCE_THRESHOLD = -20
FEED_THRESHOLD = 0.01

if not __name__ == "__main__":
    raise RuntimeError("This script should not be imported.")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dirs', nargs='+', type=str, default=["data/results/sim_results2/", "data/results/sim_results3/"],
                    help="List of directories containing simulation results")
parser.add_argument('--output_folder', type=str, default="data/results/test/preprocessed_all/",
                    help="Output folder name for processed results")
parser.add_argument('--filter_resonances', action='store_true', default=False,
                    help="Whether to filter resonances")
parser.add_argument('--filter_feed', action='store_true', default=False,
                    help="Whether to filter feed positions")
args = parser.parse_args()

all_design_params = []
all_freq_responses = []

for data_dir in args.data_dirs:
    design_params, freq_response = load_results(os.path.join(data_dir, "s_parameters"))
    sort_inds = np.argsort(design_params[:, 0])
    design_params = design_params[sort_inds]
    freq_response = freq_response[sort_inds]
    
    if args.filter_resonances:
        resonance_mask = np.any(freq_response[:, :, 1] < RESONANCE_THRESHOLD, axis=1)
        design_params = design_params[resonance_mask]
        freq_response = freq_response[resonance_mask]
    
    if args.filter_feed:
        feed_mask = np.abs(design_params[:, 2]) > FEED_THRESHOLD
        design_params = design_params[feed_mask]
        freq_response = freq_response[feed_mask]
    
    all_design_params.append(design_params)
    all_freq_responses.append(freq_response)

design_params = np.vstack(all_design_params)
freq_response = np.vstack(all_freq_responses)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
    
np.save(os.path.join(args.output_folder, "design_params.npy"), design_params)
np.save(os.path.join(args.output_folder, "freq_response.npy"), freq_response)