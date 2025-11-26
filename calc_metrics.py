import argparse
import os

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from src.metrics.utils import pesq, si_sdr_i, si_snr_i, stoi


def calculate_metrics(gt_s1_path, gt_s2_path, pred_s1_path, pred_s2_path, mix_path):
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
    mix_files = sorted(os.listdir(mix_path))
    pred_s1_files = sorted(os.listdir(pred_s1_path))
    pred_s2_files = sorted(os.listdir(pred_s2_path))

    assert (
        gt_s1_files == gt_s2_files == pred_s1_files == pred_s2_files == mix_files
    ), "File lists must match"

    for file in tqdm(gt_s1_files, desc="Calculating Metrics..."):
        gt_s1, sr1 = sf.read(os.path.join(gt_s1_path, file))
        gt_s2, sr2 = sf.read(os.path.join(gt_s2_path, file))
        mix, sr5 = sf.read(os.path.join(mix_path, file))
        pred_s1, sr3 = sf.read(os.path.join(pred_s1_path, file))
        pred_s2, sr4 = sf.read(os.path.join(pred_s2_path, file))
        gt_s1, gt_s2, mix, pred_s1, pred_s2 = [
            torch.from_numpy(audio) for audio in [gt_s1, gt_s2, mix, pred_s1, pred_s2]
        ]
        # Ensure same sample rate
        assert sr1 == sr2 == sr3 == sr4 == sr5, "Sample rates must match"

        # Calculate metrics
        stoi_score = (stoi(sr1)(pred_s1, gt_s1) + stoi(sr2)(pred_s2, gt_s2)) / 2
        pesq_score = (pesq(sr1)(pred_s1, gt_s1) + pesq(sr2)(pred_s2, gt_s2)) / 2
        si_sdr_i_score = (
            si_sdr_i(pred_s1, gt_s1, mix) + si_sdr_i(pred_s2, gt_s2, mix)
        ) / 2
        si_snr_i_score = (
            si_snr_i(pred_s1, gt_s1, mix) + si_snr_i(pred_s2, gt_s2, mix)
        ) / 2

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
    parser.add_argument("--mix", required=True, help="Path to predicted mix directory.")

    args = parser.parse_args()

    results = calculate_metrics(
        args.gt_s1, args.gt_s2, args.pred_s1, args.pred_s2, args.mix
    )
    print("Average Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
