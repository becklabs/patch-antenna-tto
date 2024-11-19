import json
import pandas as pd
import numpy as np

from deepad.simulation.manager import SweepManager

def generate_patch_configs():
    lengths_small = np.arange(7.5, 20, 1)    # 5-20mm in 1mm steps
    lengths_large = np.arange(20, 52.5, 2.5)  # 20-50mm in 2.5mm steps
    lengths = np.concatenate([lengths_small, lengths_large])
    
    wl_ratios = np.array([0.8, 1.2, 1.6, 2.0])
    
    configs = []
    
    for L in lengths:
        widths = L * wl_ratios
        
        # More dense sampling near the edge
        feeds_dense = np.linspace(0, L/6, 6) * -1  # 6 points in first sixth
        feeds_sparse = np.linspace(L/6, L/3, 6)[1:] * -1  # 5 more points up to L/3
        feed_positions = np.concatenate([feeds_dense, feeds_sparse])
        
        for W in widths:
            for feed_pos in feed_positions:
                configs.append({
                    'length_mm': round(L, 3),
                    'width_mm': round(W, 3),
                    'feed_position_mm': round(feed_pos, 3),
                })
    
    df = pd.DataFrame(configs)

    df["substrate_epsR"] = 3.68
    df["substrate_thickness"] = 1.61

    df["pulse_f0"] = 5.5e9
    df["pulse_fc"] = 4.5e9

    df["freq_start"] = 1e9
    df["freq_stop"] = 10e9
    df["n_freq"] = 1000

    df = df.sort_values(['length_mm', 'width_mm', 'feed_position_mm'])
    
    return df

if __name__ == "__main__":
    df_configs = generate_patch_configs()

    print(f"Total number of configurations: {len(df_configs)}")
    print("\nParameter ranges:")
    for col in ['length_mm', 'width_mm', 'feed_position_mm']:
        print(f"{col}:")
        print(f"  Min: {df_configs[col].min():.3f}")
        print(f"  Max: {df_configs[col].max():.3f}")

    simulator = SweepManager(configs=df_configs, sim_path="data/simulations/test2", base_dir="data/results/sim_results2")
    
    print("Initial status:")
    print(json.dumps(simulator.get_simulation_status(), indent=2))
    
    simulator.run_simulations(batch_size=10)
    
    print("\nFinal status:")
    print(json.dumps(simulator.get_simulation_status(), indent=2))