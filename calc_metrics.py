import argparse
import os

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from src.metrics.utils import pesq, si_sdr_i, si_snr_i, stoi


def calculate_metrics(gt_s1_path, gt_s2_path, pred_s1_path, pred_s2_path):
    """
    Calculate SI-SDR, SI-SIR, SI-SAR for each utterance.

    Args:
        gt_s1_path (str): Path to ground truth s1 directory.
        gt_s2_path (str): Path to ground truth s2 directory.
        pred_s1_path (str): Path to predicted s1 directory.
        pred_s2_path (str): Path to predicted s2 directory.

    Returns:
        dict: Dictionary with average metrics.
    """
    metrics = {
        "STOI": [],
        "PESQ": [],
        "SI-SDR-I": [],
        "SI-SNR-I": [],
    }

    # Assume files are named the same in all directories
    gt_s1_files = sorted(os.listdir(gt_s1_path))
    gt_s2_files = sorted(os.listdir(gt_s2_path))
    pred_s1_files = sorted(os.listdir(pred_s1_path))
    pred_s2_files = sorted(os.listdir(pred_s2_path))

    assert (
        gt_s1_files == gt_s2_files == pred_s1_files == pred_s2_files
    ), "File lists must match"

    for file in tqdm(gt_s1_files, desc="Calculating Metrics..."):
        gt_s1, sr1 = sf.read(os.path.join(gt_s1_path, file))
        gt_s2, sr2 = sf.read(os.path.join(gt_s2_path, file))
        pred_s1, sr3 = sf.read(os.path.join(pred_s1_path, file))
        pred_s2, sr4 = sf.read(os.path.join(pred_s2_path, file))

        # Ensure same sample rate
        assert sr1 == sr2 == sr3 == sr4, "Sample rates must match"

        # Stack references and estimates
        references = torch.from_numpy(np.stack([gt_s1, gt_s2], axis=0))
        mix = torch.from_numpy(
            np.stack([gt_s1 + gt_s2, gt_s1 + gt_s2], axis=0)
        )  # this need change
        estimates = torch.from_numpy(np.stack([pred_s1, pred_s2], axis=0))

        # Calculate metrics
        stoi_score = stoi(sr2)(references, estimates)
        pesq_score = pesq(sr1)(references, estimates)
        si_sdr_i_score = si_sdr_i(references, estimates, mix)
        si_snr_i_score = si_snr_i(references, estimates, mix)

        metrics["STOI"].append(stoi_score)
        metrics["PESQ"].append(pesq_score)
        metrics["SI-SDR-I"].append(si_sdr_i_score)
        metrics["SI-SNR-I"].append(si_snr_i_score)

    # Compute averages
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate audio separation metrics.")
    parser.add_argument(
        "--gt_s1", required=True, help="Path to ground truth s1 directory."
    )
    parser.add_argument(
        "--gt_s2", required=True, help="Path to ground truth s2 directory."
    )
    parser.add_argument(
        "--pred_s1", required=True, help="Path to predicted s1 directory."
    )
    parser.add_argument(
        "--pred_s2", required=True, help="Path to predicted s2 directory."
    )

    args = parser.parse_args()

    results = calculate_metrics(args.gt_s1, args.gt_s2, args.pred_s1, args.pred_s2)
    print("Average Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
