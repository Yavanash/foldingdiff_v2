"""
Secondary-structure label utilities.

3-state DSSP {H, E, C} + a PAD token. SS labels are aligned 1:1 to the residue
sequence used by foldingdiff.datasets (i.e., to the 6-angle per-residue tensor).

DSSP is computed via biotite (preferred) with a fallback to mdtraj. If neither
is installed, dssp_three_state() raises ImportError with install instructions.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import numpy as np

# Vocabulary indices. Pad is 0 so torch.zeros() defaults are safe.
SS_VOCAB = {"PAD": 0, "H": 1, "E": 2, "C": 3}
SS_VOCAB_SIZE = len(SS_VOCAB)

# DSSP 8-state to 3-state mapping. Standard: H,G,I -> H ; E,B -> E ; T,S, ,- -> C.
_DSSP_8_TO_3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", " ": "C", "-": "C", "C": "C", "P": "C",
}


def dssp_three_state(pdb_path: str) -> str:
    """
    Run DSSP on a PDB and return a 3-state SS string of length n_residues.
    Characters are 'H', 'E', or 'C'. Indexing matches the order of residues
    in the PDB chain (single-chain assumption, matching CathCanonicalAnglesDataset).
    """
    try:
        import biotite.structure.io.pdb as pdb_io
        import biotite.application.dssp as dssp_app
    except ImportError as e:
        raise ImportError(
            "ss_scaffold needs biotite for DSSP. Install with: pip install biotite"
        ) from e

    structure = pdb_io.PDBFile.read(pdb_path).get_structure(model=1)
    app = dssp_app.DsspApp(structure)
    app.start()
    app.join()
    eight_state = app.get_sse()  # numpy array of single-char codes
    three_state = "".join(_DSSP_8_TO_3.get(c, "C") for c in eight_state)
    return three_state


def encode_ss(ss_string: str, pad_to: Optional[int] = None) -> np.ndarray:
    """
    Map an 'HEC...' string to an int64 numpy array of vocabulary indices.
    Optionally right-pad with PAD (0) to length pad_to.
    """
    arr = np.array([SS_VOCAB.get(c, SS_VOCAB["C"]) for c in ss_string], dtype=np.int64)
    if pad_to is not None and pad_to > len(arr):
        out = np.zeros(pad_to, dtype=np.int64)
        out[: len(arr)] = arr
        return out
    return arr


def find_ss_runs(ss_string: str, target: str, min_len: int = 6) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of `target` SS class with length >= min_len.
    Returns list of (start, end) half-open intervals.

    Example: find_ss_runs("CCCHHHHHHHCCCEEEECC", "H", 5) -> [(3, 10)]
    """
    runs = []
    i, n = 0, len(ss_string)
    while i < n:
        if ss_string[i] == target:
            j = i
            while j < n and ss_string[j] == target:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def sample_motif_span(
    ss_string: str,
    seq_len: int,
    classes: Tuple[str, ...] = ("H", "E"),
    min_len: int = 6,
    p_no_motif: float = 0.3,
    mode: str = "ss_run",
    max_len: Optional[int] = None,
    max_flank: int = 10,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[int, int, str]]:
    """
    Sample a motif span (start, end, ss_class) from the SS string.

    Modes:
      - "ss_run": pick a random contiguous run of one of `classes` whose length
        is >= min_len. ss_class is that class's letter (legacy behavior).
      - "arbitrary_span": pick an arbitrary contiguous span [s, e) of length
        in [min_len, max_len]. ss_class is the *majority* DSSP class of the
        span (purely descriptive; the model reads per-residue ss_labels).
      - "flanks": sample (k_left, k_right) uniformly in [0, max_flank], set
        motif = [k_left, seq_len - k_right]. Mirrors the canonical inference
        case where the user gives a whole PDB and wants a few extra residues
        added at each end. Falls back to None if the resulting motif is
        shorter than min_len.
      - "mixed": uniformly pick among {"ss_run", "arbitrary_span", "flanks"}.
        Trains the model on single-SS motifs, mixed-SS chunks, AND the
        whole-protein-as-motif regime — covering the full inference range.

    Returns None with probability p_no_motif so the model keeps an
    unconditional path. Spans are bounded by seq_len.
    """
    rng = rng or random
    if rng.random() < p_no_motif:
        return None

    if mode == "mixed":
        mode = rng.choice(["ss_run", "arbitrary_span", "flanks"])

    if mode == "ss_run":
        candidates: List[Tuple[int, int, str]] = []
        for cls in classes:
            for s, e in find_ss_runs(ss_string[:seq_len], cls, min_len=min_len):
                candidates.append((s, e, cls))
        if not candidates:
            return None
        return rng.choice(candidates)

    if mode == "arbitrary_span":
        if seq_len < min_len:
            return None
        upper = seq_len if max_len is None else min(max_len, seq_len)
        if upper < min_len:
            return None
        span_len = rng.randint(min_len, upper)
        start = rng.randint(0, seq_len - span_len)
        end = start + span_len
        # Majority class is descriptive only; model uses per-residue ss_labels.
        sub = ss_string[start:end]
        majority = max("HEC", key=lambda c: sub.count(c))
        return (start, end, majority)

    if mode == "flanks":
        if seq_len < min_len:
            return None
        # Cap each flank so motif length stays >= min_len even at worst case.
        upper_flank = max(0, min(max_flank, (seq_len - min_len) // 2))
        if upper_flank < 0:
            return None
        k_left = rng.randint(0, upper_flank)
        k_right = rng.randint(0, upper_flank)
        start = k_left
        end = seq_len - k_right
        if end - start < min_len:
            return None
        sub = ss_string[start:end]
        majority = max("HEC", key=lambda c: sub.count(c))
        return (start, end, majority)

    raise ValueError(f"Unknown sample_motif_span mode: {mode!r}")


def motif_mask_from_span(
    span: Optional[Tuple[int, int, str]], pad_len: int
) -> np.ndarray:
    """
    Build an int64 (pad_len,) array with 1s at motif positions, 0 elsewhere.
    """
    mask = np.zeros(pad_len, dtype=np.int64)
    if span is not None:
        s, e, _ = span
        mask[s:e] = 1
    return mask


if __name__ == "__main__":
    # Manual sanity: run DSSP on a sample PDB, print the SS string.
    import sys

    if len(sys.argv) > 1:
        ss = dssp_three_state(sys.argv[1])
        print(f"length={len(ss)}")
        print(ss)
        for cls in "HEC":
            runs = find_ss_runs(ss, cls, min_len=6)
            print(f"{cls} runs (>=6): {runs}")
