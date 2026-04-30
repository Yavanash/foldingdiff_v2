"""Diversity & novelty metrics over a set of designable backbones."""
from __future__ import annotations

import itertools
from typing import List, Optional, Sequence

import numpy as np

from foldingdiff.tmalign import run_tmalign


def pairwise_tm(pdb_paths: Sequence[str]) -> np.ndarray:
    """Symmetric pairwise TM-score matrix. NaN on the diagonal."""
    n = len(pdb_paths)
    M = np.full((n, n), np.nan)
    for i, j in itertools.combinations(range(n), 2):
        s = run_tmalign(pdb_paths[i], pdb_paths[j], fast=True)
        M[i, j] = M[j, i] = s
    return M


def cluster_count(tm_matrix: np.ndarray, threshold: float = 0.5) -> int:
    """Greedy single-linkage clustering at TM > threshold; returns # clusters."""
    n = tm_matrix.shape[0]
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(tm_matrix[i, j]) and tm_matrix[i, j] > threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(n)})


def max_tm_to_set(query_pdb: str, reference_pdbs: Sequence[str]) -> float:
    """Novelty: 1 - max TM over a reference set (e.g. training PDBs).

    Returns the max TM (call site computes 1 - x if it wants novelty)."""
    if not reference_pdbs:
        return float("nan")
    scores = [run_tmalign(query_pdb, ref, fast=True) for ref in reference_pdbs]
    return float(np.nanmax(scores))
