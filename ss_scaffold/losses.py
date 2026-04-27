"""
Ramachandran-region prior loss.

Penalizes predicted (phi, psi) for falling outside the canonical basin for the
residue's SS class. Implemented as a negative-log-likelihood under a small
Gaussian mixture per class:

    H (alpha-helix):    single Gaussian at (-60deg, -45deg)
    E (beta-sheet):     single Gaussian at (-120deg, +130deg)
    C (coil/loop):      broad uniform-ish prior (constant), gives ~0 penalty

This is intentionally minimal: real Ramachandran maps have richer topology
(left-handed alpha basin, polyproline II, etc.) but for an academic baseline
the three principal regions are enough.

ASCII Ramachandran plot (psi up, phi right):

    +180 |       . E .                    |
         |      . . . .                   |
         |     . basin .                  |
       0 |. . . . . . . . . . . . . . . . |
         |        . H .                   |
         |       . basin                  |
    -180 |________________________________|
         -180          0          +180  phi

The H Gaussian is at (-60, -45). The E Gaussian is at (-120, +130).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

# Class indices match SS_VOCAB in ss_labels.py: PAD=0, H=1, E=2, C=3.
# Means in radians.
_PI = math.pi
_DEG = _PI / 180.0

_BASIN_MEANS = torch.tensor(
    [
        [0.0, 0.0],          # PAD: unused, contributes zero loss
        [-60.0, -45.0],      # H
        [-120.0, 130.0],     # E
        [0.0, 0.0],          # C: mean unused, treated as flat
    ],
    dtype=torch.float32,
) * _DEG

# Per-class Gaussian std in radians. Helix tighter than sheet; coil very broad.
_BASIN_LOG_STD = torch.tensor(
    [
        [0.0, 0.0],          # PAD
        [math.log(20 * _DEG), math.log(20 * _DEG)],  # H: ~20 deg
        [math.log(30 * _DEG), math.log(30 * _DEG)],  # E: ~30 deg
        [math.log(60 * _DEG), math.log(60 * _DEG)],  # C: very broad
    ],
    dtype=torch.float32,
)

# Whether each class contributes to the loss (PAD does not, C contributes
# weakly via the broad std but we zero it out for clarity).
_CLASS_ACTIVE = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)


def _angular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle from b to a, in [-pi, pi]. Handles wrap correctly."""
    return torch.remainder(a - b + _PI, 2 * _PI) - _PI


def ramachandran_loss(
    phi: torch.Tensor,
    psi: torch.Tensor,
    ss_labels: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        phi, psi:    (batch, seq_len) predicted angles in radians
        ss_labels:   (batch, seq_len) int64 in {0,1,2,3} (PAD/H/E/C)
        valid_mask:  (batch, seq_len) bool/0-1, optional. Positions with 0 are
                     excluded (e.g., padding, motif residues).
    Returns:
        scalar loss = mean over valid positions of the per-class NLL.
    """
    assert phi.shape == psi.shape == ss_labels.shape

    means = _BASIN_MEANS.to(phi.device)         # (4, 2)
    log_stds = _BASIN_LOG_STD.to(phi.device)    # (4, 2)
    active = _CLASS_ACTIVE.to(phi.device)       # (4,)

    # Per-position basin means and stds via gather on class index.
    # Shape: (batch, seq_len, 2)
    pos_means = means[ss_labels]
    pos_log_stds = log_stds[ss_labels]
    pos_active = active[ss_labels]

    d_phi = _angular_diff(phi, pos_means[..., 0])
    d_psi = _angular_diff(psi, pos_means[..., 1])

    # Gaussian NLL up to constants (constants drop out under mean over valid).
    # NLL = 0.5 * ((d/std)^2) + log_std.
    inv_std_phi = torch.exp(-pos_log_stds[..., 0])
    inv_std_psi = torch.exp(-pos_log_stds[..., 1])
    nll = 0.5 * ((d_phi * inv_std_phi) ** 2 + (d_psi * inv_std_psi) ** 2) \
        + pos_log_stds[..., 0] + pos_log_stds[..., 1]

    weight = pos_active
    if valid_mask is not None:
        weight = weight * valid_mask.to(weight.dtype)

    denom = weight.sum().clamp_min(1.0)
    return (nll * weight).sum() / denom


if __name__ == "__main__":
    # Manual heatmap check: print loss values on a coarse grid for each SS class.
    # If the basins are correct, helix should have its minimum near (-60, -45)
    # and sheet near (-120, 130). Run with: python -m ss_scaffold.losses
    import numpy as np

    grid = torch.tensor(np.linspace(-_PI, _PI, 13), dtype=torch.float32)
    for cls_name, cls_idx in [("H", 1), ("E", 2)]:
        print(f"\n{cls_name} basin loss (rows=phi deg, cols=psi deg):")
        header = "       " + " ".join(f"{int(p / _DEG):>5d}" for p in grid)
        print(header)
        for phi_v in grid:
            row = []
            for psi_v in grid:
                phi_t = phi_v.view(1, 1)
                psi_t = psi_v.view(1, 1)
                lab = torch.tensor([[cls_idx]])
                loss = ramachandran_loss(phi_t, psi_t, lab).item()
                row.append(f"{loss:>5.1f}")
            print(f"{int(phi_v / _DEG):>5d}: " + " ".join(row))
