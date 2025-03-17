import argparse
import logging
import os

import numpy as np
import torch

from patchtto.nn.datasets import RectangularPatchDataset
from patchtto.nn.decoder import ConvDecoder, SimpleFeedForwardDecoder
from patchtto.nn.encoder import ConvEncoder, SimpleFeedForwardEncoder
from patchtto.nn.losses import NLLHead
from patchtto.nn.utils import (load_checkpoint, load_config, load_data,
                             set_device)
from patchtto.nn.vae import VAE, AdversarialVAE
from patchtto.search.criterion import OracleDesignScorer, SurogateDesignScorer
from patchtto.search.initialization import (ClosestCurveInitialization,
                                          FixedRandomInitialization,
                                          KClosestCurveInitialization,
                                          RandomInitialization)
from patchtto.search.routines import find_curves, generate_design
from patchtto.signal import generate_s11_curve
from patchtto.simulation.harness import RectangularPatchHarness

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the full antenna design framework"
    )
    parser.add_argument(
        "--experiment_config",
        default="config/framework/experiment.yaml",
        help="Path to the experiment configuration file",
    )

    return parser.parse_args()

def create_s11_vae(config, device):
    encoder = ConvEncoder(latent_dim=config["model"]["latent_dim"])
    decoder = ConvDecoder(
        latent_dim=config["model"]["latent_dim"],
        output_length=config["model"]["n_freqs"],
    )
    vae = VAE(
        encoder=encoder, decoder=decoder, latent_dim=config["model"]["latent_dim"]
    )

    checkpoint_path = os.path.join(
        config["checkpoint"]["checkpoint_dir"],
        config["checkpoint"]["resume_checkpoint"],
    )
    vae, _, _, curve_scaler, _ = load_checkpoint(checkpoint_path, vae, None, device)
    vae.eval()
    return curve_scaler, vae.to(device)


def create_design_vae(config, device):
    condition_head = ConvEncoder(
        latent_dim=int(config["model"]["decoder"]["condition_features"] / 2),
    )
    encoder = SimpleFeedForwardEncoder(
        x_dim=config["model"]["n_design_params"],
        latent_dim=config["model"]["latent_dim"] * 2,
    )
    decoder = SimpleFeedForwardDecoder(
        latent_dim=config["model"]["latent_dim"]
        + config["model"]["decoder"]["condition_features"],
        output_length=config["model"]["n_design_params"],
    )
    discriminator = ConvDecoder(
        latent_dim=config["model"]["latent_dim"],
        output_length=config["model"]["n_freqs"],
        output_channels=1,
        transpose=True,
    )
    cvae = AdversarialVAE(
        encoder=encoder,
        condition_head=condition_head,
        decoder=decoder,
        discriminator=discriminator,
        latent_dim=config["model"]["latent_dim"],
    )
    checkpoint_path = os.path.join(
        config["checkpoint"]["checkpoint_dir"],
        config["checkpoint"]["resume_checkpoint"],
    )
    cvae, _, design_scaler, curve_scaler, _ = load_checkpoint(checkpoint_path, cvae, None, device)
    cvae.eval()
    return design_scaler, curve_scaler, cvae.to(device)


def create_surrogate(config, device):
    decoder = ConvDecoder(
        latent_dim=config["model"]["n_design_params"],
        output_length=config["model"]["n_freqs"],
        output_channels=2,  # mean and variance
        transpose=False
    )
    model = torch.nn.Sequential(decoder, NLLHead())
    checkpoint_path = os.path.join(
        config["checkpoint"]["checkpoint_dir"],
        config["checkpoint"]["resume_checkpoint"],
    )
    model, _, design_scaler, curve_scaler, _ = load_checkpoint(checkpoint_path, model, None, device)
    model.eval()
    return design_scaler, curve_scaler, model.to(device)


def get_init_strategy(strategy_name, s11_vae, target_curve, dataset):
    if strategy_name == "random":
        return RandomInitialization()
    elif strategy_name == "fixed_random":
        return FixedRandomInitialization()
    elif strategy_name == "closest":
        return ClosestCurveInitialization(
            vae=s11_vae,
            target_curve=target_curve,
            dataset=dataset,
        )
    elif strategy_name == "k_closest":
        return KClosestCurveInitialization(
            vae=s11_vae,
            target_curve=target_curve,
            dataset=dataset,
        )
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy_name}")


def run_experiment(
    simulation_config_path,
    design_cvae_config_path,
    s11_vae_config_path,
    s11_search_config_path,
    surrogate_config_path,
    init_strategy_name="k_closest",
    n_curves=None,
    n_steps=None,
    lr=None,
    optimize_design=False,
    n_designs=5,
    scorer_type="surrogate",
    resonant_freqs=[2.4e9],
    bandwidths=[100e6],
    depths_db=[-15],
):
    # Load configs
    s11_vae_config = load_config(s11_vae_config_path)
    s11_search_config = load_config(s11_search_config_path)
    design_cvae_config = load_config(design_cvae_config_path)
    surrogate_config = load_config(surrogate_config_path)

    device = set_device(s11_vae_config["device"])
    design_params, freq_response = load_data(s11_vae_config)

    # Override parameters if provided
    if n_curves is None:
        n_curves = s11_search_config["hyperparameters"]["n_curves"]
    if n_steps is None:
        n_steps = s11_search_config["hyperparameters"]["n_steps"]
    if lr is None:
        lr = s11_search_config["hyperparameters"]["lr"]

    freqs = np.linspace(
        float(s11_search_config["data"]["freq_start"]),
        float(s11_search_config["data"]["freq_stop"]),
        int(s11_search_config["data"]["n_freqs"]),
    )

    s11_vae_scaler, s11_vae = create_s11_vae(s11_vae_config, device)
    design_cvae_scaler, s11_cvae_scaler, design_cvae = create_design_vae(
        design_cvae_config, device
    )

    # Surrogate is only needed if we are using it
    if scorer_type in ["surrogate", "both"]:
        surrogate_design_scaler, surrogate_curve_scaler, surrogate = create_surrogate(surrogate_config, device)
    else:
        surrogate, surrogate_design_scaler, surrogate_curve_scaler = None, None, None

    dataset = RectangularPatchDataset(
        design_params=design_params,
        s11_curves=freq_response[:, :, 1],
        s11_curves_scaler=s11_vae_scaler,
        curves_device=device,
        design_device=device,
    )

    # Generate target curve with given parameters
    target_curve = generate_s11_curve(
        freq_range=freqs,
        resonant_freqs=resonant_freqs,
        bandwidths=bandwidths,
        depths_db=depths_db,
    )

    # Initialization strategy
    z_init_strategy = get_init_strategy(init_strategy_name, s11_vae, target_curve, dataset)

    # Find candidate curves
    curve_results = find_curves(
        vae=s11_vae,
        ideal_curve=target_curve,
        curve_scaler=s11_vae_scaler,
        latent_dim=s11_vae_config["model"]["latent_dim"],
        device=device,
        z_init_strategy=z_init_strategy,
        n_curves=n_curves,
        n_steps=n_steps,
        lr=lr,
        telemetry=False,
    )

    # Initialize scorers based on scorer_type
    if scorer_type in ["surrogate", "both"]:
        surrogate_design_scorer = SurogateDesignScorer(
            target_curve=target_curve,
            surrogate=surrogate,
            design_scaler=surrogate_design_scaler,
            curve_scaler=surrogate_curve_scaler,
            device=device,
        )
    else:
        surrogate_design_scorer = None

    if scorer_type in ["oracle", "both"]:
        harness = RectangularPatchHarness.from_yaml(simulation_config_path)
        oracle_design_scorer = OracleDesignScorer(
            target_curve=target_curve,
            sim_harness=harness,
        )
    else:
        oracle_design_scorer = None

    # For each generated curve, generate n_designs and score them
    design_pool = []
    for curve_info in curve_results:
        candidate_curve = curve_info["curve"]
        for _ in range(n_designs):
            design, z_star, telemetry_dict = generate_design(
                cvae=design_cvae,
                latent_dim=design_cvae_config["model"]["latent_dim"],
                candidate_curve=candidate_curve,
                optimize=optimize_design,
                design_scaler=design_cvae_scaler,
                s11_scaler=s11_cvae_scaler,
                device=device,
            )

            # Score designs
            scores = {}
            if surrogate_design_scorer is not None:
                scores["surrogate_score"] = surrogate_design_scorer(design)
            if oracle_design_scorer is not None:
                scores["oracle_score"] = oracle_design_scorer(design)

            design_pool.append((candidate_curve, design, scores))

    # Determine sorting key based on available scores
    if scorer_type in ["surrogate", "both"]:
        sort_key = lambda x: x[2].get("surrogate_score", float('inf'))
    else:
        # If only oracle is used, sort by oracle score
        sort_key = lambda x: x[2].get("oracle_score", float('inf'))

    # Sort the entire pool by the chosen metric
    design_pool_sorted = sorted(design_pool, key=sort_key)
    return design_pool_sorted


if __name__ == "__main__":
    args = parse_arguments()
    experiment_config = load_config(args.experiment_config)

    results = run_experiment(
        simulation_config_path=experiment_config["paths"]["simulation_config"],
        design_cvae_config_path=experiment_config["paths"]["design_cvae_config"],
        s11_vae_config_path=experiment_config["paths"]["s11_vae_config"],
        s11_search_config_path=experiment_config["paths"]["s11_search_config"],
        surrogate_config_path=experiment_config["paths"]["surrogate_config"],
        init_strategy_name=experiment_config["experiment"]["init_strategy"],
        n_curves=experiment_config["experiment"]["n_curves"],
        n_steps=experiment_config["experiment"]["n_steps"],
        lr=experiment_config["experiment"]["lr"],
        optimize_design=experiment_config["experiment"]["optimize_design"],
        n_designs=experiment_config["experiment"]["n_designs"],
        scorer_type=experiment_config["experiment"]["scorer_type"],
        resonant_freqs=[float(f) for f in experiment_config["target_curve"]["resonant_freqs"]],
        bandwidths=[float(b) for b in experiment_config["target_curve"]["bandwidths"]],
        depths_db=[float(d) for d in experiment_config["target_curve"]["depths_db"]]
    )

    for candidate_curve, design, scores in results:
        print("Design:", design)
        print("Scores:", scores)
        print("-" * 50)
    
    # 
