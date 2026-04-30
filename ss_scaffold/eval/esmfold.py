"""ESMFold wrapper: sequence -> predicted PDB + per-residue pLDDT.

Loads the model once; call `fold(seq)` for each sequence.
Requires:  pip install fair-esm[esmfold]  (or the HF transformers EsmForProteinFolding).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


class ESMFoldRunner:
    def __init__(self, device: Optional[str] = None, chunk_size: int = 64):
        try:
            import esm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ESMFold requires `fair-esm[esmfold]`. "
                "Install with: pip install 'fair-esm[esmfold]'"
            ) from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading ESMFold v1 on {self.device} (this is slow first time)")
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().to(self.device)
        # chunk_size trades speed for memory; lower it on small GPUs.
        self.model.set_chunk_size(chunk_size)

    @torch.no_grad()
    def fold(self, seq: str) -> Tuple[str, np.ndarray]:
        """Fold a single sequence; return (pdb_string, per_residue_plddt)."""
        out = self.model.infer([seq])
        pdb_str = self.model.output_to_pdb(out)[0]
        # ESMFold returns pLDDT per atom; collapse to per-residue by mean over CA mask.
        plddt = out["plddt"][0].detach().cpu().numpy()  # (L, 37) atomic pLDDT
        # Atom 1 is CA in atom37; that's the standard summary.
        plddt_per_res = plddt[:, 1]
        return pdb_str, plddt_per_res

    def fold_to_file(self, seq: str, out_pdb: str) -> Tuple[str, np.ndarray]:
        pdb_str, plddt = self.fold(seq)
        Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
        Path(out_pdb).write_text(pdb_str)
        return out_pdb, plddt


def fold_many(seqs: List[str], out_dir: str, runner: Optional[ESMFoldRunner] = None,
              prefix: str = "fold") -> List[Tuple[str, np.ndarray]]:
    runner = runner or ESMFoldRunner()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, s in enumerate(seqs):
        pdb_path = out_dir / f"{prefix}_{i:03d}.pdb"
        try:
            _, plddt = runner.fold_to_file(s, str(pdb_path))
            results.append((str(pdb_path), plddt))
        except Exception as e:  # noqa: BLE001
            logging.warning(f"ESMFold failed on seq {i}: {e}")
            results.append((None, None))
    return results
