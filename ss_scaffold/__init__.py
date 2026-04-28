"""
ss_scaffold: secondary-structure-conditioned motif scaffolding on top of FoldingDiff.

Reuses foldingdiff.modelling.BertForDiffusion as the backbone, adds:
  - 3-state SS embedding ({H, E, C}) injected additively at the input layer
  - is_motif channel concatenated to the 6 backbone angles (n_inputs: 6 -> 7)
  - Ramachandran-region Gaussian-mixture loss over (phi, psi) per SS class
  - RePaint-style motif inpainting at sampling time

Designed for an academic coursework project (BT305). See README.md.
"""

from ss_scaffold.ss_labels import SS_VOCAB, dssp_three_state, sample_motif_span
from ss_scaffold.losses import ramachandran_loss
from ss_scaffold.model import BertForSSConditionedDiffusion
from ss_scaffold.sampling import p_sample_loop_with_motif

__all__ = [
    "SS_VOCAB",
    "dssp_three_state",
    "sample_motif_span",
    "ramachandran_loss",
    "BertForSSConditionedDiffusion",
    "p_sample_loop_with_motif",
]
