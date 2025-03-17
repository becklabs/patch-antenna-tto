import os
import numpy as np
from typing import Tuple


def build_FDTD(
    patch_width: float = 40,
    patch_length: float = 32,

    substrate_epsR: float = 3.38,

    substrate_width: float = 60,
    substrate_length: float = 60, 
    substrate_thickness: float = 1.524,
    substrate_cells: int = 4,

    feed_R: float = 50,
    feed_pos: Tuple[float, float] = (0,-8),

    SimBox: np.ndarray = np.array([200, 200, 150]),

    f0: float = 2e9,
    fc: float = 1e9,
):
    """
    Build and configure an FDTD simulation for a microstrip patch antenna using openEMS.
    
    This function creates the FDTD object and sets up the complete simulation geometry
    but does not run the simulation.

    Parameters
    ----------
    patch_width : float, optional
        Width [x] of the patch antenna in millimeters (default: 40)
    patch_length : float, optional
        Length [y] of the patch antenna in millimeters (default: 32)
    substrate_epsR : float, optional
        Relative permittivity of the substrate material (default: 3.38)
    substrate_width : float, optional
        Width [x] of the substrate in millimeters (default: 60)
    substrate_length : float, optional
        Length [y] of the substrate in millimeters (default: 60)
    substrate_thickness : float, optional
        Thickness in z of the substrate in millimeters (default: 1.524)
    substrate_cells : int, optional
        Number of mesh cells across substrate thickness (default: 4)
    feed_R : float, optional
        Feed port resistance in ohms (default: 50)
    feed_pos : Tuple[float, float], optional
        XY-position of the feed point in millimeters from center (default: (0, -8))
    SimBox : np.ndarray, optional
        Simulation box dimensions [x, y, z] in millimeters (default: [200, 200, 150])
    f0 : float, optional
        Center frequency in Hz for Gaussian excitation (default: 2e9)
    fc : float, optional
        Cutoff frequency in Hz for Gaussian excitation (default: 1e9)

    Returns
    -------
    tuple
        (FDTD, port), where:
        - FDTD: configured openEMS FDTD object ready for simulation
        - port: configured port object for post-processing

    Notes
    -----
    - All spatial dimensions are in millimeters
    - The antenna is centered at (0,0) in the xy-plane
    - Uses MUR absorbing boundary conditions
    - Mesh resolution is automatically set to λ/20 at (f0 + fc)
    """
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS.physical_constants import EPS0, C0
    
    substrate_kappa = 1e-3 * 2 * np.pi * 2.45e9 * EPS0 * substrate_epsR

    # Create FDTD
    FDTD = openEMS(NrTS=30000, EndCriteria=1e-3)
    # FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    # FDTD.SetBoundaryCond(['PML_8'] * 6)
    FDTD.SetBoundaryCond(['MUR'] * 6)

    # Create CSX
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    mesh_res = C0 / (f0 + fc) / 1e-3 / 20 #
    # mesh_res = C0 / (f0 + fc) / 1e-3 / 40 #

    # Initialize the mesh with the "air-box" dimensions
    mesh.AddLine('x', [-SimBox[0] / 2, SimBox[0] / 2])
    mesh.AddLine('y', [-SimBox[1] / 2, SimBox[1] / 2])
    mesh.AddLine('z', [-SimBox[2] / 3, SimBox[2] * 2 / 3])

    # Create patch
    patch = CSX.AddMetal('patch')  # PEC
    start = [-patch_width / 2, -patch_length / 2, substrate_thickness]
    stop = [patch_width / 2, patch_length / 2, substrate_thickness]
    patch.AddBox(priority=10, start=start, stop=stop)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res / 2)

    # Create substrate
    substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
    start = [-substrate_width / 2, -substrate_length / 2, 0]
    stop = [substrate_width / 2, substrate_length / 2, substrate_thickness]
    substrate.AddBox(priority=0, start=start, stop=stop)

    # Add extra cells to discretize the substrate thickness
    mesh.AddLine('z', np.linspace(0, substrate_thickness, substrate_cells + 1))

    # Create ground (same size as substrate)
    gnd = CSX.AddMetal('gnd')  # PEC
    start[2] = 0
    stop[2] = 0
    gnd.AddBox(start, stop, priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

    # Apply the excitation & resistor as a current source
    start = [feed_pos[0], feed_pos[1], 0]
    stop = [feed_pos[0], feed_pos[1], substrate_thickness]
    print(start, stop)
    port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0, priority=5, edges2grid='xy')

    mesh.SmoothMeshLines('all', mesh_res, 1.4)

    return FDTD, port

def run_FDTD(
    FDTD,
    simulation_path,
    cleanup: bool = True,
    verbose: int = 3
) -> str:
    """
    Run a configured FDTD simulation.

    Parameters
    ----------
    FDTD : openEMS FDTD object
        Configured FDTD object from build_FDTD
    simulation_path : str
        Path to store simulation files
    cleanup : bool, optional
        Whether to clean up temporary files after simulation (default: True)
    verbose : int, optional
        Verbosity level (0-3) (default: 3)

    Returns
    -------
    str
        Absolute path where simulation files are stored
    """
    simulation_path = os.path.abspath(simulation_path)
    curr_dir = os.getcwd()
    FDTD.Run(simulation_path, cleanup=cleanup, verbose=verbose)
    os.chdir(curr_dir)
    return simulation_path

def compute_S11(
    port,
    simulation_path: str,
    freqs: np.ndarray
):
    """
    Process FDTD simulation results to calculate S-parameters.

    Parameters
    ----------
    port : openEMS port object
        Port object returned from setup_and_run_FDTD
    simulation_path : str
        Path where simulation files are stored
    freqs : np.ndarray
        Frequency points for S-parameter calculation

    Returns
    -------
    tuple
        (frequencies, s11_dB) where frequencies are the frequency points in Hz
        and s11_dB is the S11 parameter in dB
    """
    port.CalcPort(simulation_path, freqs)
    s11 = port.uf_ref / port.uf_inc
    s11_dB = 20.0 * np.log10(np.abs(s11))
    return s11_dB





