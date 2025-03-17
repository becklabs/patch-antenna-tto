import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from scripts.framework.run import run_experiment

target_curves = [
    {"resonant_freqs": [2.4e9], "bandwidths": [100e6], "depths_db": [-15]},
    {"resonant_freqs": [3.5e9], "bandwidths": [200e6], "depths_db": [-15]},
    {"resonant_freqs": [8.0e9], "bandwidths": [100e6], "depths_db": [-15]},
]

init_strategy = "k_closest"  
n_curves = 1                  
optimize_values = [False, True]
n_designs_list = [1, 3, 5, 10, 20]
scorer_type = "surrogate"

simulation_config_path = "config/simulation/rectangular_patch.yaml"
design_cvae_config_path = "config/train/design_cvae.yaml"
s11_vae_config_path = "config/train/s11_vae.yaml"
s11_search_config_path = "config/search/search_s11.yaml"
surrogate_config_path = "config/train/surrogate_nll.yaml"

results_file = "data/experiments/design_sampling_experiment_results.json"
os.makedirs(Path(results_file).parent, exist_ok=True)

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        all_results = json.load(f)
else:
    all_results = {}

    for tc_index, tc in enumerate(target_curves):
        tc_key = f"target_curve_{tc_index}"
        all_results[tc_key] = {
            "resonant_freqs": tc["resonant_freqs"],
            "bandwidths": tc["bandwidths"],
            "depths_db": tc["depths_db"],
            "data": {}
        }
        for optimize_design in optimize_values:
            opt_key = "optimize_true" if optimize_design else "optimize_false"
            all_results[tc_key]["data"][opt_key] = {}
            for n_designs in n_designs_list:
                print(f"Running: TargetCurve={tc['resonant_freqs'][0]}Hz, Optimize={optimize_design}, n_designs={n_designs}")
                results = run_experiment(
                    simulation_config_path=simulation_config_path,
                    design_cvae_config_path=design_cvae_config_path,
                    s11_vae_config_path=s11_vae_config_path,
                    s11_search_config_path=s11_search_config_path,
                    surrogate_config_path=surrogate_config_path,
                    init_strategy_name=init_strategy,
                    n_curves=n_curves,
                    n_steps=None,   
                    lr=None,        
                    optimize_design=optimize_design,
                    n_designs=n_designs,
                    scorer_type=scorer_type,
                    resonant_freqs=tc["resonant_freqs"],
                    bandwidths=tc["bandwidths"],
                    depths_db=tc["depths_db"]
                )

                scores = [r[2].get("surrogate_score", float('inf')) for r in results]
                lowest_score = min(scores) if scores else float('inf')

                all_results[tc_key]["data"][opt_key][str(n_designs)] = lowest_score

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

# We want to plot the average lowest score vs n_designs for both optimize_design=True and False.

# Initialize structures to store results
optimize_labels = {False: r"Not Optimized ($z_x$ fixed)", True: r"Optimized ($z_x$ variable)"}
optimize_keys = ["optimize_false", "optimize_true"]

# Collect scores from all target curves
strategy_scores = {ok: {nd: [] for nd in n_designs_list} for ok in optimize_keys}

for tc_key in all_results.keys():
    tc_data = all_results[tc_key]
    for opt_key in optimize_keys:
        for nd in n_designs_list:
            score = tc_data["data"][opt_key][str(nd)]
            strategy_scores[opt_key][nd].append(score)

plt.figure(figsize=(8, 4.8))
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16
})

colors = ['#4B0082', '#008B8B']  # Deep purple and teal
markers = ['o', 'o']

for idx, opt_key in enumerate(optimize_keys):
    means = []
    stds = []
    for nd in n_designs_list:
        scores = strategy_scores[opt_key][nd]
        means.append(np.mean(scores))
        stds.append(np.std(scores))

    means = np.array(means)
    stds = np.array(stds)

    plt.plot(n_designs_list, means,
             color=colors[idx],
             marker=markers[idx],
             markersize=8,
             label=optimize_labels[(opt_key == "optimize_true")],
             clip_on=False)
    plt.fill_between(n_designs_list,
                     means - stds,
                     means + stds,
                     color=colors[idx],
                     alpha=0.15)

plt.xlabel("Number of Designs")
plt.ylabel("Average Lowest Surrogate Score")

plt.grid(True, linestyle=':', alpha=0.3, zorder=0)

plt.legend(loc='upper right',
           frameon=True,
           edgecolor='none')

plt.tight_layout(rect=[0, 0, 0.7, 1])

plot_filename = "figs/experiments/n_designs_scaling.pdf"
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_filename}")

# Also save as PNG
plt.savefig(plot_filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')

plt.show()
