"""
Sampling entrypoint for ss_scaffold.

Given:
  - a trained ss_scaffold checkpoint dir
  - a motif specified as a PDB file + residue range (e.g. "10-30")
  - a target total length

Generates a backbone of length L where residues [motif_start:motif_end] are
clamped to the motif's true angles and the rest are denoised by the model.

Outputs reconstructed PDB files via foldingdiff.angles_and_coords.

Usage:
  python -m ss_scaffold.sample \\
      --model-dir runs/ss_scaffold_v1 \\
      --motif-pdb data/example_helix.pdb \\
      --motif-range 10-30 \\
      --total-length 80 \\
      --n-samples 8 \\
      --out generated/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from foldingdiff import angles_and_coords as ac
from foldingdiff import beta_schedules, utils
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals

from ss_scaffold.model import BertForSSConditionedDiffusion
from ss_scaffold.sampling import p_sample_loop_with_motif
from ss_scaffold.ss_labels import (
    SS_VOCAB,
    dssp_three_state,
    encode_ss,
)


# Match the model's default schema (canonical-full-angles).
ANGLE_NAMES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
IS_ANGLE = [True] * 6


def _parse_range(s: str) -> Tuple[int, int]:
    a, b = s.split("-")
    return int(a), int(b)


def _extract_motif_angles(pdb_path: str, start: int, end: int) -> np.ndarray:
    """Extract the 6 backbone angles for residues [start:end] from a PDB."""
    df = canonical_distances_and_dihedrals(pdb_path, angles=ANGLE_NAMES)
    if df is None or len(df) < end:
        raise ValueError(
            f"Could not extract angles {start}:{end} from {pdb_path} "
            f"(got {0 if df is None else len(df)} residues)"
        )
    return df.iloc[start:end][ANGLE_NAMES].to_numpy(dtype=np.float32)


def _build_ss_string(
    total_length: int,
    motif_start: int,
    motif_end: int,
    motif_ss: str,
    flank_ss: str = "C",
) -> str:
    """
    Build the per-residue SS string fed to the model.

    motif_ss: per-residue SS for the motif region (length must equal
              motif_end - motif_start). Comes from DSSP on the input PDB,
              so mixed-SS inputs are represented faithfully.
    flank_ss: single character ('H', 'E', or 'C') used for every residue
              outside the motif region. Default 'C' biases the flanks toward
              loops; pass 'H' or 'E' to ask for helical/sheet padding.
    """
    if len(motif_ss) != motif_end - motif_start:
        raise ValueError(
            f"motif_ss length {len(motif_ss)} != motif span "
            f"{motif_end - motif_start}"
        )
    if flank_ss not in ("H", "E", "C"):
        raise ValueError(f"flank_ss must be one of H/E/C, got {flank_ss!r}")
    chars = [flank_ss] * total_length
    for i, c in enumerate(motif_ss):
        chars[motif_start + i] = c
    return "".join(chars)


def _full_pdb_residue_count(pdb_path: str) -> int:
    df = canonical_distances_and_dihedrals(pdb_path, angles=ANGLE_NAMES)
    if df is None:
        raise ValueError(f"Could not parse {pdb_path}")
    return len(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Dir saved by training (config.json + checkpoint).")
    p.add_argument("--checkpoint", default=None, help="Specific .ckpt path; else use model-dir/checkpoints/last.ckpt")
    p.add_argument("--motif-pdb", required=True)
    p.add_argument("--motif-range", default=None,
                   help="e.g. 10-30 (half-open, 0-indexed). Defaults to the "
                        "whole input PDB, i.e. the motif IS the input.")
    p.add_argument("--motif-class", default=None,
                   help="Optional override: force every motif residue to this "
                        "SS class (H/E/C). By default per-residue DSSP labels "
                        "are used so mixed-SS inputs are handled correctly.")
    p.add_argument("--flank-ss", default="C", choices=["H", "E", "C"],
                   help="SS class assigned to the residues added before/after "
                        "the motif. Default 'C' (loop-biased flanks).")
    p.add_argument("--total-length", type=int, required=True)
    p.add_argument("--motif-target-start", type=int, default=None,
                   help="Where to place the motif in the generated protein. "
                        "Defaults to the same start index as in the source PDB.")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-schedule", default="cosine")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.motif_range is None:
        motif_start_src = 0
        motif_end_src = _full_pdb_residue_count(args.motif_pdb)
        logging.info(f"--motif-range not given; using full input PDB "
                     f"[0, {motif_end_src})")
    else:
        motif_start_src, motif_end_src = _parse_range(args.motif_range)
    motif_len = motif_end_src - motif_start_src
    motif_target_start = (
        args.motif_target_start
        if args.motif_target_start is not None
        else motif_start_src
    )
    motif_target_end = motif_target_start + motif_len
    assert motif_target_end <= args.total_length, "Motif overruns total length"

    logging.info(f"Extracting motif from {args.motif_pdb}: residues {motif_start_src}-{motif_end_src}")
    motif_angles_np = _extract_motif_angles(args.motif_pdb, motif_start_src, motif_end_src)

    means_path = Path(args.model_dir) / "training_means.npy"
    if means_path.exists():
        training_means = np.load(means_path).astype(np.float32)
        assert training_means.shape == (len(ANGLE_NAMES),), (
            f"training_means shape {training_means.shape} != ({len(ANGLE_NAMES)},)"
        )
        logging.info(f"Loaded training_means: {training_means}")
        motif_angles_np = utils.modulo_with_wrapped_range(
            motif_angles_np - training_means, range_min=-np.pi, range_max=np.pi
        )
    else:
        training_means = None
        logging.warning(
            f"No training_means.npy found at {means_path}; sampling in raw angle space. "
            "If the model was trained with zero_center=True, outputs will be biased."
        )

    # Per-residue SS for the motif region. Use DSSP unless the user forces a
    # single class via --motif-class.
    if args.motif_class is not None:
        if args.motif_class not in ("H", "E", "C"):
            raise ValueError(f"--motif-class must be H/E/C, got {args.motif_class!r}")
        motif_ss = args.motif_class * motif_len
        logging.info(f"Motif SS forced to '{args.motif_class}' x {motif_len}")
    else:
        try:
            ss_src = dssp_three_state(args.motif_pdb)
            motif_ss = ss_src[motif_start_src:motif_end_src]
            if len(motif_ss) != motif_len:
                raise ValueError(
                    f"DSSP returned {len(ss_src)} residues but motif range is "
                    f"[{motif_start_src},{motif_end_src})"
                )
            logging.info(f"Motif DSSP labels: {motif_ss}")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"DSSP failed ({e}); falling back to all-coil motif labels.")
            motif_ss = "C" * motif_len

    # Load model.
    model = BertForSSConditionedDiffusion.from_dir(args.model_dir, load_weights=False)
    ckpt_path = args.checkpoint or str(Path(args.model_dir) / "checkpoints" / "last.ckpt")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model = model.to(args.device).eval()

    # Build per-sample inputs.
    L = args.total_length
    B = args.n_samples
    n_features = len(ANGLE_NAMES)

    ss_string = _build_ss_string(
        L, motif_target_start, motif_target_end, motif_ss, flank_ss=args.flank_ss,
    )
    ss_labels = torch.from_numpy(encode_ss(ss_string)).long().unsqueeze(0).expand(B, -1).contiguous()

    motif_mask = torch.zeros((B, L), dtype=torch.long)
    motif_mask[:, motif_target_start:motif_target_end] = 1

    motif_angles = torch.zeros((B, L, n_features), dtype=torch.float32)
    motif_t = torch.from_numpy(motif_angles_np).unsqueeze(0).expand(B, -1, -1)
    motif_angles[:, motif_target_start:motif_target_end, :] = motif_t

    # Initial noise: angle-bounded random in [-pi, pi].
    noise = (torch.rand(B, L, n_features) * 2 - 1) * float(np.pi)

    betas = beta_schedules.get_variance_schedule(args.beta_schedule, args.timesteps)

    logging.info(f"Sampling {B} backbones of length {L} with motif at "
                 f"[{motif_target_start},{motif_target_end}) "
                 f"motif_ss={motif_ss} flank_ss={args.flank_ss}")

    sampled = p_sample_loop_with_motif(
        model=model,
        lengths=[L] * B,
        noise=noise,
        timesteps=args.timesteps,
        betas=betas,
        ss_labels=ss_labels,
        motif_mask=motif_mask,
        motif_angles=motif_angles,
        is_angle=IS_ANGLE,
    )
    final = sampled[-1].numpy()  # (B, L, n_features)

    if training_means is not None:
        final = utils.modulo_with_wrapped_range(
            final + training_means, range_min=-np.pi, range_max=np.pi
        )

    # Save per-sample CSV (angles) and reconstruct PDBs via NeRF.
    meta = {
        "model_dir": args.model_dir,
        "motif_pdb": args.motif_pdb,
        "motif_range": [motif_start_src, motif_end_src],
        "motif_ss": motif_ss,
        "flank_ss": args.flank_ss,
        "motif_class_override": args.motif_class,
        "motif_target": [motif_target_start, motif_target_end],
        "total_length": L,
        "n_samples": B,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    for i in range(B):
        df = pd.DataFrame(final[i], columns=ANGLE_NAMES)
        df.to_csv(out_dir / f"sample_{i:03d}.csv", index=False)
        try:
            ac.create_new_chain_nerf(str(out_dir / f"sample_{i:03d}.pdb"), df)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"NeRF reconstruction failed for sample {i}: {e}")

    logging.info(f"Wrote {B} samples to {out_dir}")


if __name__ == "__main__":
    main()
