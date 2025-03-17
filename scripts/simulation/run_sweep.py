import json
import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay, KDTree
from mpl_toolkits.mplot3d import Axes3D
import random

from patchtto.simulation.manager import SweepManager
from patchtto.simulation.utils import load_results


def sample_points_within_hull(hull, num_samples):
    """
    Samples random points within the convex hull using rejection sampling.

    Parameters:
    - hull: scipy.spatial.ConvexHull object representing the convex hull.
    - num_samples: Number of points to sample.

    Returns:
    - points: NumPy array of shape (num_samples, 3) with sampled points.
    """
    # Get the bounding box of the hull
    min_bounds = np.min(hull.points, axis=0)
    max_bounds = np.max(hull.points, axis=0)

    points = []
    delaunay = Delaunay(hull.points[hull.vertices])

    attempts = 0
    max_attempts = num_samples * 100  

    while len(points) < num_samples and attempts < max_attempts:
        random_point = np.random.uniform(min_bounds, max_bounds)
        if delaunay.find_simplex(random_point) >= 0:
            points.append(random_point)
        attempts += 1

    if len(points) < num_samples:
        print(f"Warning: Only {len(points)} points were sampled within the convex hull out of requested {num_samples}.")

    return np.array(points)

def farthest_point_sampling(existing_points, hull, num_new_points, num_candidates=1000):
    """
    Performs farthest point sampling to add new points within the convex hull.

    Parameters:
    - existing_points: NumPy array of shape (M, 3) with existing data points.
    - hull: scipy.spatial.ConvexHull object representing the convex hull.
    - num_new_points: Number of new points to add.
    - num_candidates: Number of candidate points to sample at each iteration.

    Returns:
    - new_points: NumPy array of shape (num_new_points, 3) with new points.
    """
    new_points = []

    for i in range(num_new_points):
        # Sample candidate points within the hull
        candidates = sample_points_within_hull(hull, num_candidates)

        if candidates.size == 0:
            print("No more candidate points can be sampled within the hull.")
            break

        # For each candidate, find the distance to the nearest existing or new point
        if new_points:
            combined_points = np.vstack([existing_points, new_points])
        else:
            combined_points = existing_points

        tree_combined = KDTree(combined_points)
        distances, _ = tree_combined.query(candidates, k=1)

        # Select the candidate with the maximum minimum distance
        max_dist_idx = np.argmax(distances)
        best_candidate = candidates[max_dist_idx]
        best_distance = distances[max_dist_idx]

        new_points.append(best_candidate)

        if (i + 1) % 10 == 0 or i == num_new_points - 1:
            print(f"Added {i + 1}/{num_new_points} new points. Current max distance: {best_distance:.3f}")

    return np.array(new_points)

def augment_patch_configs(num_points=1000, data_dir="data/results/sim_results2/s_parameters"):
    np.random.seed(42)
    random.seed(42)

    all_points, freq_response = load_results(data_dir)
    resonance_mask = np.any(freq_response[:, :, 1] < -20, axis = 1)
    existing_points = all_points[resonance_mask]

    hull = ConvexHull(existing_points)

    num_new_points = num_points

    # Perform Farthest Point Sampling to add new points
    new_points = farthest_point_sampling(existing_points, hull, num_new_points, num_candidates=1000)

    configs = []
    for point in new_points:
        configs.append({
            'length_mm': round(point[0], 3),
            'width_mm': round(point[1], 3),
            'feed_position_mm': round(point[2], 3),
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
    results_dir = "data/results/sim_results3"
    sim_path = "data/simulations/test3"
    df_configs = augment_patch_configs(num_points=1000)
    # df_configs = generate_patch_configs()

    print(f"Total number of configurations: {len(df_configs)}")
    print("\nParameter ranges:")
    for col in ['length_mm', 'width_mm', 'feed_position_mm']:
        print(f"{col}:")
        print(f"  Min: {df_configs[col].min():.3f}")
        print(f"  Max: {df_configs[col].max():.3f}")

    simulator = SweepManager(configs=df_configs, sim_path=sim_path, base_dir=results_dir)
    
    print("Initial status:")
    print(json.dumps(simulator.get_simulation_status(), indent=2))
    
    simulator.run_simulations(batch_size=10)
    
    print("\nFinal status:")
    print(json.dumps(simulator.get_simulation_status(), indent=2))
