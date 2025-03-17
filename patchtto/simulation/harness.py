import logging
import tempfile
from abc import ABC, abstractmethod

import numpy as np
import yaml

logger = logging.getLogger(__name__)
from .openems import build_FDTD, compute_S11, run_FDTD


class SimulationHarness(ABC):
    """
    Base class for simulation harnesses
    """

    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    @abstractmethod
    def simulate(self, designs: np.ndarray) -> np.ndarray:
        """
        Simulate a batch of designs and return the S11 (abs) response
        """
        ...


class RectangularPatchHarness(SimulationHarness):
    """
    Ephemeral simulation harness for rectangular patch antennas

                    width
                <---------->
                ┌──────────┐ ↑
                │          │ │
                │          │ │
                │    ●     │ │ length
       feed_pos │    │     │ │
                │    │     │ │
                │    ↓     │ │
                └──────────┘ ↓

    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.substrate_factor = float(self.config["substrate"]["substrate_factor"])
        self.substrate_epsR = float(self.config["substrate"]["substrate_epsR"])
        self.substrate_thickness = float(
            self.config["substrate"]["substrate_thickness"]
        )

        self.airbox_factor = float(self.config["airbox"]["airbox_factor"])
        self.airbox_thickness = float(self.config["airbox"]["airbox_thickness"])

        self.pulse_f0 = float(self.config["frequency"]["pulse_f0"])
        self.pulse_fc = float(self.config["frequency"]["pulse_fc"])
        self.freq_start = float(self.config["frequency"]["freq_start"])
        self.freq_stop = float(self.config["frequency"]["freq_stop"])
        self.n_freqs = int(self.config["frequency"]["n_freqs"])

        self.freqs = np.linspace(self.freq_start, self.freq_stop, self.n_freqs)

    def simulate(self, designs: np.ndarray) -> np.ndarray:
        assert (
            designs.shape[1] == 3
        ), "Designs must have 3 columns: width, length, feed_pos"
        s11 = np.empty((designs.shape[0], self.n_freqs))
        for i, design in enumerate(designs):
            s11[i] = self._simulate_single(*design)
        return s11

    def _simulate_single(
        self, length: float, width: float, feed_pos: float
    ) -> np.ndarray:
        """
        Simulate a single design and return the S11 (abs) response
        """
        box_size = int(self.airbox_factor * max(width, length))
        substrate_width = int(self.substrate_factor * max(width, length))
        substrate_length = substrate_width
        SimBox = np.array([box_size, box_size, self.airbox_thickness])

        try:
            fdtd, port = build_FDTD(
                patch_width=width,
                patch_length=length,
                substrate_epsR=self.substrate_epsR,
                substrate_thickness=self.substrate_thickness,
                substrate_width=substrate_width,
                substrate_length=substrate_length,
                SimBox=SimBox,
                feed_pos=(0, feed_pos),
                f0=self.pulse_f0,
                fc=self.pulse_fc,
            )

            with tempfile.TemporaryDirectory() as sim_path:
                path = run_FDTD(fdtd, sim_path, verbose=0)
                s11 = compute_S11(port, path, self.freqs)
                return s11
        except Exception as e:
            logger.error(
                "Error simulating design (%f, %f, %f): %s", length, width, feed_pos, e
            )
            raise e
