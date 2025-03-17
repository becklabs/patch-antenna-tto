# Improving Generative Inverse Design of Rectangular Patch Antennas with Test Time Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) [![arXiv](https://img.shields.io/badge/arXiv-2303.00000-b31b1b.svg)](https://arxiv.org/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)

---

## Overview
This repository contains the code and experiments described in our paper **"Improving Generative Inverse Design of Rectangular Patch Antennas with Test Time Optimization."**. We propose a two-stage inverse design framework for generating rectangular patch antennas that meet target frequency response specifications. Further, we show that leveraging search and optimization techniques at test-time improves the accuracy of the generated designs and enables consideration of auxiliary
objectives such as manufacturability.

<div align="center">
<p align="center">
  <img src="figs/paper/framework.png" width="600px" alt="diagram">
</p>
</div>

The framework leverages generative deep learning models combined with test time optimization to efficiently perform inverse design of rectangular patch antennas. By integrating simulation, surrogate modeling, and optimization, the approach improves design quality and accelerates the overall design process.


The repository implements a complete end-to-end framework for generative inverse design of rectangular patch antennas. Key components include:

- **Generative Models:** Variational Autoencoders (VAEs) and Conditional VAEs (CVAEs) for learning low-dimensional representations of antenna design parameters and corresponding frequency responses (S11 curves).
- **Test Time Optimization:** Optimization routines that refine latent vectors during inference to better match target S11 responses.
- **Simulation Harness:** A simulation pipeline based on openEMS for evaluating the performance of generated designs.
- **Surrogate Models:** Forward models that predict S11 responses and enable fast scoring during design search.
- **Design Search & Experimentation:** Scripts to run inverse design experiments, study scaling with number of curves/designs, and compare surrogate vs. oracle scoring.

This modular framework enables both rapid prototyping and rigorous evaluation, facilitating improvements in generative inverse design as presented in our paper.

---

## Setup

#### Install from Source

1. **Clone the repository and navigate into its directory:**
    
    ```bash
    git clone https://github.com/becklabs/patch-antenna-tto.git
    cd patch-antenna-tto
    ```
    
2. **Install the `patchtto` package in editable mode:**
    
    ```bash
    pip install -e .

    ```

#### Install openEMS (optional)
To run the simulation harness, you will need to install `openEMS`. Detailed instructions can be found [here](https://openems.com/docs/install/). `openEMS` was successfully installed on Apple M2 via the following method:

1. Update Homebrew
```bash
brew update
brew upgrade
brew cleanup
```

2. Tap the openEMS repository
```bash
brew tap thliebig/openems https://github.com/thliebig/openEMS-Project.git
```

3. Install openEMS
```bash
brew install --HEAD openems
```

4. Build the CSXCAD Python bindings
```bash
cd ~/Library/Caches/Homebrew/openems--git/CSXCAD/python
python setup.py build_ext -I /opt/homebrew/opt/openems/include -L /opt/homebrew/opt/openems/lib -R /opt/homebrew/opt/openems/lib
```

5. Build the openEMS Python bindings
```bash
cd ~/Library/Caches/Homebrew/openems--git/openEMS/python
python setup.py build_ext -I /opt/homebrew/opt/openems/include -L /opt/homebrew/opt/openems/lib -R /opt/homebrew/opt/openems/lib
```


#### Login to Weights & Biases (optional)
For tracking training experiments, you will need a [Weights & Biases](https://wandb.ai/site) account:

```bash
wandb login
```


### Data Artifacts

This project uses simulation and preprocessed data for training and evaluation. To run experiments:

1. **Download the Preprocessed Data:** Download the provided `data_artifacts.zip` (or follow the instructions in the paper’s supplementary materials) and extract it to the `data/` directory at the repository root.
    
    ```bash
    unzip /path/to/data_artifacts.zip -d data/
    ```
    
2. **(Optional) Run the Simulation Preprocessing:** If you wish to generate your own simulation data, use the preprocessing script:
    
    ```bash
    python scripts/simulation/preprocess.py --data_dirs data/results/sim_results2/ data/results/sim_results3/ --output_folder data/results/preprocessed_all_filtered_feed/
    ```
    

---

## Training

The repository contains several training scripts for different components of the framework:

### Design CVAE Training

Train the Conditional VAE (CVAE) model for antenna design:

```bash
python scripts/design_cvae/train.py --config config/train/design_cvae.yaml
```

### S11 VAE Training

Train the S11 VAE to learn representations of antenna frequency responses:

```bash
python scripts/s11_vae/train.py --config config/train/s11_vae.yaml
```

### Forward Model Training (Beta-NLL Loss)

Train the forward surrogate model for predicting S11 curves:

```bash
python scripts/forward_model/train_betanll.py --config config/train/surrogate_nll.yaml
```

Each configuration file under the `config/` directory specifies training hyperparameters, dataset paths, and checkpointing details. 

---

## Experiments

Reproduce experiments from the paper using the provided scripts:

### Inverse Design Experiment

Run the full inverse design framework to generate antenna designs that meet target S11 specifications:

```bash
python scripts/experiments/inverse_design.py
```

This script combines generative modeling, latent space optimization, and simulation-based scoring to produce candidate designs.

### Scaling Experiments

Study the effect of varying the number of latent curves or design samples:

- **Number of Curves Scaling:**
    
    ```bash
    python scripts/experiments/n_curves_scaling.py
    ```
    
- **Number of Designs Sampling:**
    
    ```bash
    python scripts/experiments/n_designs_scaling.py
    ```
    

### Simulation Sweep

Perform a simulation sweep over a range of rectangular patch configurations:

```bash
python scripts/simulation/run_sweep.py
```

This script uses farthest point sampling to generate new design points within the convex hull of existing simulation data.

---

## Citation

If you find this work useful, please consider citing our paper:

> **Improving Generative Inverse Design of Rectangular Patch Antennas with Test Time Optimization**  
> _Authors et al._, [Conference/Journal Name, Year]

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://chatgpt.com/c/LICENSE) file for details.

---

## Contact

For questions or feedback, please open an issue or contact the maintainers at [your-email@example.com](mailto:your-email@example.com).

---

Happy designing!