import os

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi


def create_and_upload_dataset(
    dataset_dir="data/results/preprocessed_all",
    readme_path="README.md",
    figure_paths=["patch_antenna_diagram.png", "s11_example.png"],
    repo_id=None,  # Format: "username/dataset_name"
    private=False,
    token=None     # Hugging Face API token
):
    """
    Create a dataset from antenna data and upload it to the Hugging Face Hub
    along with a README and figure files.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory where the dataset files are stored
    readme_path : str
        Path to the README.md file
    figure_paths : list
        List of paths to the figure files
    repo_id : str
        Hugging Face repository ID (username/dataset_name)
    token : str
        Hugging Face API token
    """
    design_params_file = os.path.join(dataset_dir, "design_params.npy")
    design_params = np.load(design_params_file)
    lengths = design_params[:, 0]
    widths = design_params[:, 1]
    feed_ys = design_params[:, 2]

    freq_response_file = os.path.join(dataset_dir, "freq_response.npy")
    freq_response = np.load(freq_response_file)
    freqs = freq_response[:, :, 0]
    s11 = freq_response[:, :, 1]

    dataset_dict = {
        "length": lengths.tolist(),
        "width": widths.tolist(),
        "feed_y": feed_ys.tolist(),
        "frequencies": freqs.tolist(),
        "s11": s11.tolist(),
        "id": list(range(len(lengths)))
    }

    features = Features({
        "length": Value("float32"),
        "width": Value("float32"),
        "feed_y": Value("float32"),
        "frequencies": Sequence(Value("float32")),
        "s11": Sequence(Value("float32")),
        "id": Value("int32")
    })

    dataset = Dataset.from_dict(dataset_dict, features=features)


    print(f"Created dataset with {len(dataset)} samples")
    
    if repo_id is None or "/" not in repo_id:
        raise ValueError("Please provide a valid repo_id in the format 'username/dataset_name'")
    
    print(f"\nUploading dataset to {repo_id}...")
    
    commit_message = "Upload rectangular patch antenna dataset"
    
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        commit_message=commit_message,
        private=private,  
        embed_external_files=False  
    )
    
    api = HfApi()
    
    if os.path.exists(readme_path):
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
    else:
        print(f"Warning: README file not found at {readme_path}")
    
    for fig_path in figure_paths:
        if os.path.exists(fig_path):
            fig_name = os.path.basename(fig_path)
            print(f"Uploading {fig_name} to assets folder...")
            assets_path = f"assets/{fig_name}"
            api.upload_file(
                path_or_fileobj=fig_path,
                path_in_repo=assets_path,  
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
        else:
            print(f"Warning: Figure file not found at {fig_path}")
            
    print("\nUpload completed successfully!")
    print(f"Your dataset is now available at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not found.")
        print("Please set it before running this script:")
        print("export HF_TOKEN=your_token_here")
        import sys
        sys.exit(1)
    
    your_username = "becklabash"
    dataset_name = "rectangular-patch-antenna-freq-response"
    repo_id = f"{your_username}/{dataset_name}"
    
    create_and_upload_dataset(
        dataset_dir="data/results/preprocessed_all",
        readme_path="data/huggingface/DATASET_README.md",
        figure_paths=["figs/huggingface/patch_antenna_diagram.png", "figs/huggingface/s11_example.png"],
        repo_id=repo_id,
        token=hf_token
    )