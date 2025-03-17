import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging
import traceback
from typing import Dict, Any, Optional


from .openems import build_FDTD, run_FDTD, compute_S11


class SweepManager:
    def __init__(
        self, configs: pd.DataFrame, sim_path: str, base_dir: str = "simulation_results"
    ):
        """
        Initialize the simulation manager.

        Args:
            base_dir: Base directory for storing all simulation data
        """
        self.sim_path = sim_path
        self.base_dir = Path(base_dir)
        self.configs = configs
        self.progress_file = self.base_dir / "simulation_progress.json"
        self.results_dir = self.base_dir / "s_parameters"
        self.log_file = self.base_dir / "simulation.log"

        self.base_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Load or initialize progress tracking
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        """Load or initialize progress tracking."""
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            progress = {
                "completed_configs": [],
                "failed_configs": [],
                "last_index": -1,
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
            }
            self._save_progress(progress)
            return progress

    def _save_progress(self, progress: Dict[str, Any]):
        """Save current progress to file."""
        progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def _get_result_filename(self, config_id: int) -> Path:
        """Generate filename for storing simulation results."""
        return self.results_dir / f"result_{config_id:06d}.npz"

    def _simulate_single_config(
        self, config: pd.Series
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Run an FDTD sim with openEMS
        """
        try:

            patch_length = float(config["length_mm"])
            patch_width = float(config["width_mm"])
            feed_pos = float(config["feed_position_mm"])

            substrate_epsR = float(config["substrate_epsR"])
            substrate_thickness = float(config["substrate_thickness"])

            pulse_f0 = float(config["pulse_f0"])
            pulse_fc = float(config["pulse_fc"])

            box_size = int(3.5 * max(patch_width, patch_length))
            substrate_width = int(1.5 * max(patch_width, patch_length))
            substrate_length = substrate_width
            SimBox = np.array([box_size, box_size, 150])

            fdtd, port = build_FDTD(
                patch_width=patch_width,
                patch_length=patch_length,
                substrate_epsR=substrate_epsR,
                substrate_thickness=substrate_thickness,
                substrate_width=substrate_width,
                substrate_length=substrate_length,
                SimBox=SimBox,
                feed_pos=(0, feed_pos),
                f0=pulse_f0,
                fc=pulse_fc,
            )

            path = run_FDTD(fdtd, self.sim_path, verbose=0)

            freq_start = int(config["freq_start"])
            freq_stop = int(config["freq_stop"])
            n_freq = int(config["n_freq"])
            freq = np.linspace(freq_start, freq_stop, n_freq)

            s11 = compute_S11(port, path, freq)

            return {"frequency": freq, "s11": s11, "config": config.to_dict()}

        except Exception as e:
            self.logger.error("Simulation failed for config %s: %s", config.name, str(e))
            traceback.print_exc()
            return None

    def run_simulations(self, batch_size: int = 1):
        """
        Run simulations for all configurations with checkpointing.

        Args:
            batch_size: Number of simulations to run before saving progress
        """
        configs_df = self.configs
        total_configs = len(configs_df)

        self.logger.info(
            "Starting simulation batch with %d total configurations", total_configs
        )
        self.logger.info(
            "Previously completed: %d", len(self.progress["completed_configs"])
        )

        try:
            for idx, (config_id, config) in enumerate(configs_df.iterrows(), start=1):
                if config_id in self.progress["completed_configs"]:
                    continue

                if config_id in self.progress["failed_configs"]:
                    continue

                self.logger.info(
                    "Running simulation %d/%d (Config ID: %s)", 
                    idx, total_configs, config_id
                )

                result = self._simulate_single_config(config)

                if result is not None:
                    result_file = self._get_result_filename(config_id)
                    np.savez_compressed(result_file, **result)
                    self.progress["completed_configs"].append(config_id)
                else:
                    self.progress["failed_configs"].append(config_id)

                if idx % batch_size == 0:
                    self._save_progress(self.progress)
                    self.logger.info(
                        "Progress saved. Completed %d simulations",
                        len(self.progress["completed_configs"])
                    )

        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        finally:
            self._save_progress(self.progress)
            self.logger.info("Final progress saved")

    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        configs_df = self.configs
        total_configs = len(configs_df)
        completed = len(self.progress["completed_configs"])
        failed = len(self.progress["failed_configs"])

        return {
            "total_configurations": total_configs,
            "completed_simulations": completed,
            "failed_simulations": failed,
            "remaining_simulations": total_configs - completed - failed,
            "progress_percentage": (completed / total_configs) * 100,
            "start_time": self.progress["start_time"],
            "last_update": self.progress["last_update"],
        }
