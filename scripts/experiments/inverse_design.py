import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from scripts.framework.run import run_experiment
from patchtto.simulation.harness import RectangularPatchHarness

from patchtto.signal import generate_s11_curve

target_curves = [
    {"resonant_freqs": [2.4e9], "bandwidths": [100e6], "depths_db": [-15]},
    {"resonant_freqs": [4.0e9], "bandwidths": [150e6], "depths_db": [-20]},
    {"resonant_freqs": [5.0e9], "bandwidths": [300e6], "depths_db": [-10]},
]

conditions = [
    {"init_strategy": "random", "n_curves": 1, "n_designs": 1, "optimize_design": False},
    {"init_strategy": "random", "n_curves": 10, "n_designs": 20, "optimize_design": False},
]

simulation_config_path = "config/simulation/rectangular_patch.yaml"
design_cvae_config_path = "config/train/design_cvae.yaml"
s11_vae_config_path = "config/train/s11_vae.yaml"
s11_search_config_path = "config/search/search_s11.yaml"
surrogate_config_path = "config/train/surrogate_nll.yaml"

scorer_type = "surrogate"

results_file = "data/experiments/simulation_overlay_results.json"
os.makedirs(Path(results_file).parent, exist_ok=True)


if os.path.exists(results_file):
    with open(results_file, "r") as f:
        all_results = json.load(f)
    print(f"Loaded existing results from {results_file}")
else:
    all_results = {}
    
    harness = RectangularPatchHarness.from_yaml(simulation_config_path)
    freqs = harness.freqs  

    for tc_index, tc in enumerate(target_curves):
        tc_key = f"target_curve_{tc_index}"
        all_results[tc_key] = {
            "resonant_freqs": tc["resonant_freqs"],
            "bandwidths": tc["bandwidths"],
            "depths_db": tc["depths_db"],
            "data": {}
        }

        for cond_index, cond in enumerate(conditions):
            cond_key = f"condition_{cond_index}"
            all_results[tc_key]["data"][cond_key] = {
                "init_strategy": cond["init_strategy"],
                "n_curves": cond["n_curves"],
                "n_designs": cond["n_designs"],
                "optimize_design": cond["optimize_design"]
            }

            # Run the experiment
            print(f"Running: TargetCurve={tc['resonant_freqs'][0]}Hz, "
                  f"init={cond['init_strategy']}, "
                  f"n_curves={cond['n_curves']}, "
                  f"n_designs={cond['n_designs']}, "
                  f"optimize={cond['optimize_design']}")

            results = run_experiment(
                simulation_config_path=simulation_config_path,
                design_cvae_config_path=design_cvae_config_path,
                s11_vae_config_path=s11_vae_config_path,
                s11_search_config_path=s11_search_config_path,
                surrogate_config_path=surrogate_config_path,
                init_strategy_name=cond["init_strategy"],
                n_curves=cond["n_curves"],
                n_steps=None,   # Use config default
                lr=None,        # Use config default
                optimize_design=cond["optimize_design"],
                n_designs=cond["n_designs"],
                scorer_type=scorer_type,
                resonant_freqs=tc["resonant_freqs"],
                bandwidths=tc["bandwidths"],
                depths_db=tc["depths_db"]
            )

            best_score = float('inf')
            best_design = None
            for entry in results:
                candidate_curve, design, scores = entry
                score = scores.get("surrogate_score", float('inf'))
                if score < best_score:
                    best_score = score
                    best_design = design

            if best_design is None:
                print(f"No valid designs found for {tc_key} under {cond_key}")
                continue

            if isinstance(best_design, torch.Tensor):
                best_design = best_design.detach().cpu().numpy()
            elif isinstance(best_design, list):
                best_design = np.array(best_design)
            elif isinstance(best_design, np.ndarray):
                pass
            else:
                raise TypeError(f"Unsupported design type: {type(best_design)}")

            if best_design.ndim == 1:
                best_design = best_design.reshape(1, -1)  # Shape to (1, 3)

            assert best_design.shape[1] == 3, "Design must have 3 parameters: length, width, feed_pos"

            try:
                simulated_s11 = harness.simulate(best_design)  # Shape: (1, n_freqs)
            except Exception as e:
                print(f"Simulation failed for {tc_key} under {cond_key}: {e}")
                simulated_s11 = np.full((1, len(freqs)), np.nan)  # Fill with NaNs

            all_results[tc_key]["data"][cond_key]["best_score"] = best_score
            all_results[tc_key]["data"][cond_key]["design"] = best_design.flatten().tolist()
            all_results[tc_key]["data"][cond_key]["simulated_s11_db"] = simulated_s11.flatten().tolist()

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_file}")

harness = RectangularPatchHarness.from_yaml(simulation_config_path)
freqs = harness.freqs  


os.makedirs("figs/simulation_overlays", exist_ok=True)


# Create a single figure with subplots for each target curve
n_target_curves = len(all_results)
fig, axes = plt.subplots(n_target_curves, 1, figsize=(12, 5*n_target_curves))
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16
})

# If there's only one target curve, wrap the axis in a list for consistent indexing
if n_target_curves == 1:
    axes = [axes]

# Plot each target curve and its simulations
for tc_index, (tc_key, tc_data) in enumerate(all_results.items()):
    ax = axes[tc_index]
    
    target_curve_params = {
        "resonant_freqs": tc_data["resonant_freqs"],
        "bandwidths": tc_data["bandwidths"],
        "depths_db": tc_data["depths_db"]
    }

    # Plot simulated curves for each condition first
    colors = ['#1f77b4', '#d62728']  # More distinct colors: blue and red for simulations
    for cond_index, (cond_key, cond_data) in enumerate(tc_data["data"].items()):
        simulated_s11_db = np.array(cond_data.get("simulated_s11_db", []))
        if simulated_s11_db.size == 0 or np.isnan(simulated_s11_db).all():
            print(f"Skipping plot for {tc_key} under {cond_key} due to simulation failure.")
            continue

        # label = (f"{cond_data['init_strategy'].replace('_', ' ').title()}, "
        #         f"n_curves={cond_data['n_curves']}, "
        #         f"n_designs={cond_data['n_designs']}")
        label = (f"#curves={cond_data['n_curves']}, "
                f"#designs={cond_data['n_designs']}")
        ax.plot(freqs / 1e9, simulated_s11_db, 
                label=label, 
                color=colors[cond_index],
                linewidth=1.5)

    # Generate and plot the ideal target curve last (so it's on top)
    ideal_curve = generate_s11_curve(
        freq_range=freqs,
        resonant_freqs=target_curve_params["resonant_freqs"],
        bandwidths=target_curve_params["bandwidths"],
        depths_db=target_curve_params["depths_db"],
    )
    ax.plot(freqs / 1e9, ideal_curve, 
            label="Target Curve", 
            color='black',  
            linestyle='--', 
            linewidth=2,
            zorder=10)  # Ensure it's always on top

    subtitle = (f"Target: f0={target_curve_params['resonant_freqs'][0]/1e9:.1f} GHz, "
               f"BW={target_curve_params['bandwidths'][0]/1e6:.0f} MHz, "
               f"Depth={target_curve_params['depths_db'][0]} dB")
    ax.set_title(subtitle)
    
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend()
    
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$|S_{11}|_\text{dB}$")


plt.tight_layout()

plot_filename_pdf = "figs/experiments/inverse_design.pdf"
plot_filename_png = "figs/experiments/inverse_designs.png"
plt.savefig(plot_filename_pdf, dpi=300, bbox_inches='tight')
plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to {plot_filename_pdf} and {plot_filename_png}")

plt.show()