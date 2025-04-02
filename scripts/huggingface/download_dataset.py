import os
import numpy as np
import logging
from datasets import load_dataset # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = load_dataset("becklabash/rectangular-patch-antenna-freq-response", split="train", cache_dir="data/huggingface/")
logger.info(f"Loaded dataset with {len(dataset)} rows")

def process_row(row):
    return {
        "design": np.array([row["length"], row["width"], row["feed_y"]]),
        "freq_response": np.column_stack((row["frequencies"], row["s11"])),
    }

preprocessed = dataset.map(process_row)
logger.info(f"Preprocessed dataset with {len(preprocessed)} rows")

design_params = np.array([row["design"] for row in preprocessed])
freq_response = np.array([row["freq_response"] for row in preprocessed])

output_dir = "data/results/preprocessed_all"

logger.info(f"Design params shape: {design_params.shape}")
logger.info(f"Freq response shape: {freq_response.shape}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(os.path.join(output_dir, "design_params.npy"), design_params)
np.save(os.path.join(output_dir, "freq_response.npy"), freq_response)

logger.info(f"Saved design_params and freq_response to {output_dir}")

