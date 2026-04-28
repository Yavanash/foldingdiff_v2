"""
Dataset wrapper that adds SS labels and a motif mask on top of FoldingDiff's
existing CathCanonicalAnglesDataset + NoisedAnglesDataset stack.

We do NOT subclass NoisedAnglesDataset (its __getitem__ already returns the
diffusion dict we want). Instead we wrap it: pull the dict out, attach
ss_labels, motif_mask, and motif_angles, and return.

Reuses (do not re-implement):
  - foldingdiff.datasets.CathCanonicalAnglesDataset    (raw angle features)
  - foldingdiff.datasets.NoisedAnglesDataset           (diffusion noising)
  - foldingdiff.utils.modulo_with_wrapped_range        (angular wrap)

What's new here:
  - DSSP-derived SS labels per residue, cached alongside the angle cache
  - Per-example motif span sampling (helix or sheet) with no-motif fallback
  - Motif-mask channel that signals which residues are clamped to ground truth
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from foldingdiff.datasets import CathCanonicalAnglesDataset, NoisedAnglesDataset

from ss_scaffold.ss_labels import (
    SS_VOCAB_SIZE,
    dssp_three_state,
    encode_ss,
    motif_mask_from_span,
    sample_motif_span,
)


def _ss_cache_path(cache_dir: Path) -> Path:
    """SS labels live next to the angle cache so they invalidate together."""
    return cache_dir / "ss_labels_v1.pkl"


class SSAnnotatedAnglesDataset(Dataset):
    """
    Wraps a NoisedAnglesDataset (which itself wraps a CathCanonicalAnglesDataset)
    and returns the same dict augmented with:
      - ss_labels:    (pad,) int64 in {PAD, H, E, C}
      - motif_mask:   (pad,) int64 in {0, 1}, 1 = motif residue (clamp to truth)
      - motif_angles: (pad, n_features) float32, the unnoised motif angles
                      (zeros at non-motif positions). Used for inpainting clamp
                      at sampling time and (optionally) for masking the loss.

    SS labels are computed via DSSP at construction time and cached under the
    same data dir as foldingdiff's angle cache, with a separate pickle file.
    """

    def __init__(
        self,
        wrapped: NoisedAnglesDataset,
        pdb_dir: str,
        cache_dir: Optional[str] = None,
        motif_classes: Tuple[str, ...] = ("H", "E"),
        motif_min_len: int = 6,
        p_no_motif: float = 0.3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        # Reach through the noiser to find the underlying angles dataset so we
        # can align SS labels with the residue order it uses.
        inner = wrapped.dset
        assert isinstance(inner, CathCanonicalAnglesDataset), (
            f"SSAnnotatedAnglesDataset expects an inner CathCanonicalAnglesDataset, "
            f"got {type(inner)}"
        )
        self.inner = inner
        self.pdb_dir = Path(pdb_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.pdb_dir
        self.motif_classes = tuple(motif_classes)
        self.motif_min_len = motif_min_len
        self.p_no_motif = p_no_motif
        self._rng = np.random.default_rng(seed)

        self._ss_strings: List[str] = self._load_or_compute_ss()

    def _load_or_compute_ss(self) -> List[str]:
        cache_file = _ss_cache_path(self.cache_dir)
        # Cache key includes the inner dataset's filename list so we invalidate
        # if the underlying data changes.
        key = tuple(os.path.basename(f) for f in self.inner.filenames)
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                logging.info(f"Loaded SS cache: {cache_file}")
                return blob["ss"]
            logging.info(f"SS cache key mismatch, recomputing: {cache_file}")

        logging.info(f"Computing DSSP for {len(self.inner.filenames)} structures...")
        ss_strings: List[str] = []
        for fname in self.inner.filenames:
            try:
                ss_strings.append(dssp_three_state(fname))
            except Exception as e:  # noqa: BLE001
                logging.warning(f"DSSP failed for {fname}: {e}; using all-coil.")
                # Fall back to all-coil; length unknown without parsing the PDB,
                # so use a generous upper bound and let downstream truncate.
                ss_strings.append("C" * self.wrapped.pad)

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"key": key, "ss": ss_strings}, f)
        logging.info(f"Wrote SS cache: {cache_file}")
        return ss_strings

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.wrapped[index]
        # Recover the original (un-exhaustive-t) item index for SS lookup.
        if self.wrapped.exhaustive_timesteps:
            item_index = index // self.wrapped.timesteps
        else:
            item_index = index

        pad = self.wrapped.pad
        ss_string = self._ss_strings[item_index]
        ss_labels = encode_ss(ss_string, pad_to=pad)

        # Sequence length is the count of unmasked positions in attn_mask if the
        # base dataset provides it, else fall back to len(ss_string).
        attn_mask = item.get("attn_mask")
        if attn_mask is not None:
            seq_len = int(attn_mask.sum().item())
        else:
            seq_len = min(len(ss_string), pad)

        span = sample_motif_span(
            ss_string,
            seq_len=seq_len,
            classes=self.motif_classes,
            min_len=self.motif_min_len,
            p_no_motif=self.p_no_motif,
        )
        motif_mask = motif_mask_from_span(span, pad_len=pad)

        # Pull the clean angles for the motif region and zero them elsewhere.
        # `angles` is provided by CathCanonicalAnglesDataset's __getitem__.
        angles = item["angles"]
        motif_angles = torch.zeros_like(angles)
        if span is not None:
            s, e, _ = span
            motif_angles[s:e] = angles[s:e]

        item["ss_labels"] = torch.from_numpy(ss_labels)
        item["motif_mask"] = torch.from_numpy(motif_mask)
        item["motif_angles"] = motif_angles
        return item

    # Pass-throughs so the wrapped object can substitute for NoisedAnglesDataset
    # in foldingdiff.sampling.sample()'s duck-typed interface.
    @property
    def feature_names(self):
        return self.wrapped.feature_names

    @property
    def feature_is_angular(self):
        return self.wrapped.feature_is_angular

    @property
    def pad(self):
        return self.wrapped.pad

    @property
    def timesteps(self):
        return self.wrapped.timesteps

    @property
    def alpha_beta_terms(self):
        return self.wrapped.alpha_beta_terms

    @property
    def filenames(self):
        return self.wrapped.filenames

    def sample_length(self, *args, **kwargs):
        return self.wrapped.sample_length(*args, **kwargs)

    def sample_noise(self, vals: torch.Tensor) -> torch.Tensor:
        return self.wrapped.sample_noise(vals)
