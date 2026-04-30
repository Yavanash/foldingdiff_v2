"""Per-sample metrics: scRMSD, scTM, motif-RMSD, DSSP fidelity.

Designability convention (Yim et al. 2023, FrameDiff / Lin et al. RFdiffusion):
    designable = min_over_seqs(scRMSD) < 2.0 AND mean_pLDDT > 70
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from biotite.structure.io.pdb import PDBFile

from foldingdiff.tmalign import run_tmalign


def _ca_coords(pdb_path: str, residues: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return (N, 3) CA coordinates from a PDB. `residues` is 0-indexed."""
    arr = PDBFile.read(pdb_path).get_structure(model=1)
    ca = arr[(arr.atom_name == "CA")]
    coords = ca.coord
    if residues is not None:
        coords = coords[list(residues)]
    return np.asarray(coords, dtype=np.float64)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """CA-RMSD after optimal rigid alignment (Kabsch). P, Q: (N, 3)."""
    assert P.shape == Q.shape and P.ndim == 2 and P.shape[1] == 3
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def sc_rmsd(generated_pdb: str, predicted_pdb: str) -> float:
    """Self-consistency RMSD: CA-RMSD between generated backbone and ESMFolded prediction."""
    P = _ca_coords(generated_pdb)
    Q = _ca_coords(predicted_pdb)
    if P.shape != Q.shape:
        L = min(len(P), len(Q))
        logging.warning(f"Length mismatch ({len(P)} vs {len(Q)}); truncating to {L}")
        P, Q = P[:L], Q[:L]
    return kabsch_rmsd(P, Q)


def sc_tm(generated_pdb: str, predicted_pdb: str) -> float:
    """Self-consistency TM-score (normalized by reference = generated)."""
    return float(run_tmalign(predicted_pdb, generated_pdb, fast=True))


def motif_rmsd(generated_pdb: str, predicted_pdb: str,
               motif_residues: Sequence[int]) -> float:
    """CA-RMSD restricted to motif residues, after global Kabsch on the motif slab."""
    P = _ca_coords(generated_pdb, motif_residues)
    Q = _ca_coords(predicted_pdb, motif_residues)
    return kabsch_rmsd(P, Q)


def dssp_fidelity(predicted_pdb: str, requested_ss: str) -> Tuple[float, str]:
    """Fraction of residues whose DSSP class matches the requested SS string."""
    from ss_scaffold.ss_labels import dssp_three_state
    obs = dssp_three_state(predicted_pdb)
    L = min(len(obs), len(requested_ss))
    if L == 0:
        return float("nan"), ""
    matches = sum(1 for i in range(L) if obs[i] == requested_ss[i])
    return matches / L, obs[:L]


def is_designable(min_sc_rmsd: float, mean_plddt: float,
                  rmsd_threshold: float = 2.0, plddt_threshold: float = 70.0) -> bool:
    return (min_sc_rmsd < rmsd_threshold) and (mean_plddt > plddt_threshold)
