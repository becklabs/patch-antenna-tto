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
init_strategies = ["random", "k_closest"]
init_strategy_labels = {
    "random": "Random",
    "k_closest": r"$k$-closest"
}

n_curves_list = [1, 3, 5, 10, 20]

simulation_config_path = "config/simulation/rectangular_patch.yaml"
design_cvae_config_path = "config/train/design_cvae.yaml"
s11_vae_config_path = "config/train/s11_vae.yaml"
s11_search_config_path = "config/search/search_s11.yaml"
surrogate_config_path = "config/train/surrogate_nll.yaml"

optimize_design = False
n_designs = 1
scorer_type = "surrogate"  

results_file = "data/experiments/n_curves_scaling_noweight/experiment_results.json"

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
        for init_strategy in init_strategies:
            all_results[tc_key]["data"][init_strategy] = {}
            for n_curves in n_curves_list:
                print(f"Running: TargetCurve={tc['resonant_freqs'][0]}Hz, Strategy={init_strategy}, n_curves={n_curves}")
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

                # Each element of results is (candidate_curve, design, scores)
                # We want the lowest "surrogate_score"
                # If multiple scores are present, we assume "surrogate_score" is the key
                scores = [r[2].get("surrogate_score", float('inf')) for r in results]
                lowest_score = min(scores) if scores else float('inf')

                all_results[tc_key]["data"][init_strategy][str(n_curves)] = lowest_score

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)


# Initialize dictionaries to store scores for each strategy and n_curves
strategy_scores = {strategy: {n: [] for n in n_curves_list} for strategy in init_strategies}

# Collect all scores
for tc_key in all_results.keys():
    tc_data = all_results[tc_key]
    for init_strategy in init_strategies:
        for n_curves in n_curves_list:
            score = tc_data["data"][init_strategy][str(n_curves)]
            strategy_scores[init_strategy][n_curves].append(score)

plt.figure(figsize=(8, 4.8))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16
})

# Color scheme - using distinct colors
colors = ['#4B0082', '#008B8B']  # Deep purple and teal
markers = ['o', 'o']  # Using circles for both

for idx, init_strategy in enumerate(init_strategies):
    means = []
    stds = []
    for n_curves in n_curves_list:
        scores = strategy_scores[init_strategy][n_curves]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Plot mean line with confidence interval
    plt.plot(n_curves_list, means, 
            color=colors[idx],
            marker=markers[idx],
            markersize=8,
            label=init_strategy_labels[init_strategy],
            clip_on=False)
    plt.fill_between(n_curves_list, 
                    means - stds, 
                    means + stds, 
                    color=colors[idx],
                    alpha=0.15)

plt.xlabel("Number of Curves")
plt.ylabel("Average Lowest Surrogate Score")

# Customize grid - lighter and in background
plt.grid(True, linestyle=':', alpha=0.3, zorder=0)

plt.legend(loc='upper right',
          frameon=True,
          edgecolor='none')

plt.tight_layout(rect=[0, 0, 0.7, 1])

plot_filename = "figs/experiments/n_curves_scaling.pdf"  
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_filename}")

plt.savefig(plot_filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')

plt.show()

