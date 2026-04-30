"""ProteinMPNN wrapper: backbone PDB -> N candidate sequences.

Assumes a local clone of ProteinMPNN. Path resolution order:
    1. --mpnn-script CLI flag (passed by orchestrator)
    2. $PROTEIN_MPNN_SCRIPT env var
    3. ~/software/ProteinMPNN/protein_mpnn_run.py (matches bin/ scripts)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


def resolve_mpnn_script(explicit: Optional[str] = None) -> str:
    candidates = [
        explicit,
        os.environ.get("PROTEIN_MPNN_SCRIPT"),
        os.path.expanduser("~/software/ProteinMPNN/protein_mpnn_run.py"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "ProteinMPNN script not found. Set --mpnn-script, $PROTEIN_MPNN_SCRIPT, "
        "or clone to ~/software/ProteinMPNN/."
    )


def _read_fasta(path: Path) -> List[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def design_sequences(
    pdb_path: str,
    n_sequences: int = 8,
    temperature: float = 0.1,
    ca_only: bool = True,
    seed: int = 1234,
    mpnn_script: Optional[str] = None,
) -> List[str]:
    """Run ProteinMPNN on a single PDB; return sampled sequences (no header).

    `ca_only=True` matches what foldingdiff/bin/ uses since NeRF-reconstructed
    backbones are CA-anchored. The first entry in MPNN's fasta is the input
    sequence, so we slice it off.
    """
    script = resolve_mpnn_script(mpnn_script)
    pdb_path = str(Path(pdb_path).resolve())

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cmd = [
            "python", script,
            "--pdb_path", pdb_path,
            "--pdb_path_chains", "A",
            "--out_folder", str(td),
            "--num_seq_per_target", str(n_sequences),
            "--sampling_temp", str(temperature),
            "--seed", str(seed),
            "--batch_size", str(n_sequences),
        ]
        if ca_only:
            cmd.append("--ca_only")

        rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            raise RuntimeError(f"ProteinMPNN failed (rc={rc}) on {pdb_path}")

        bname = Path(pdb_path).stem + ".fa"
        fasta = td / "seqs" / bname
        if not fasta.is_file():
            raise FileNotFoundError(f"Expected fasta at {fasta}")
        all_seqs = _read_fasta(fasta)

    # First entry is the input sequence parsed from the PDB; rest are sampled.
    sampled = all_seqs[1:]
    if len(sampled) != n_sequences:
        logging.warning(
            f"MPNN returned {len(sampled)} sampled seqs (expected {n_sequences}) for {pdb_path}"
        )
    return sampled
