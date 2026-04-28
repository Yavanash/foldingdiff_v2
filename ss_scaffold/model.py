"""
BertForSSConditionedDiffusion: SS-conditioned, motif-aware extension of
foldingdiff.modelling.BertForDiffusion.

What's new vs. parent:
  - SS embedding: nn.Embedding(SS_VOCAB_SIZE, hidden_size) added to inputs at
    the input layer, alongside the time embedding. Aligned 1:1 to residue
    positions, so additive injection is sufficient (see design doc).
  - is_motif channel: an extra scalar input feature concatenated to the 6
    angles. The internal input projection takes 7 channels, but ft_names /
    ft_is_angular stay as the 6 angle features so the parent's per-feature
    loss logging still works.
  - Motif-mask in loss: predicted noise at motif positions is excluded from the
    angular loss (the model shouldn't be penalized for "predicting" residues
    whose angle values it can read directly from input).
  - Ramachandran-region prior loss: per-non-motif residue, weighted by
    self.rama_lambda. Disabled by default (lambda=0) so this class is
    backward-equivalent to the parent until you turn it on.

Data flow:

         angles (B, L, 6)        is_motif (B, L, 1)        ss_labels (B, L)
              \\                       /                      |
               concat -----------------                       |
                       |                                      |
                       v                                      v
                 inputs_to_hidden_dim                   ss_embedding
                 (Linear 7 -> hidden)                   (nn.Embedding)
                       |                                      |
                       +-- + position_embed -- + time_embed --+
                                       |
                                       v
                                  BertEncoder
                                       |
                                       v
                                 token_decoder
                                       |
                                       v
                          predicted noise (B, L, 6)
"""
from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from transformers.optimization import get_linear_schedule_with_warmup

from foldingdiff.modelling import BertForDiffusion

from ss_scaffold.losses import ramachandran_loss
from ss_scaffold.ss_labels import SS_VOCAB_SIZE


# Default angle schema matches FoldingDiff's "canonical-full-angles" feature set.
_DEFAULT_FT_IS_ANGULAR: List[bool] = [True, True, True, True, True, True]
_DEFAULT_FT_NAMES: List[str] = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]


class BertForSSConditionedDiffusion(BertForDiffusion):
    """
    Inherits the full Lightning machinery (configure_optimizers, training_step,
    validation_step, lr schedules) from BertForDiffusion. We only override:
      - __init__: replace inputs_to_hidden_dim with one that takes n+1 inputs,
        add ss_embedding.
      - forward: fold ss_labels and is_motif into the input stream.
      - _get_loss_terms: mask motif positions and add Ramachandran prior.
    """

    def __init__(
        self,
        rama_lambda: float = 0.0,
        ss_vocab_size: int = SS_VOCAB_SIZE,
        ft_is_angular: Optional[List[bool]] = None,
        ft_names: Optional[List[str]] = None,
        pretrained_checkpoint: Optional[str] = None,
        freeze_pretrained: bool = False,
        **kwargs,
    ) -> None:
        if ft_is_angular is None:
            ft_is_angular = list(_DEFAULT_FT_IS_ANGULAR)
        if ft_names is None:
            ft_names = list(_DEFAULT_FT_NAMES)

        # Construct parent with the bare angle schema so per-feature loss
        # labelling (in training_step / validation_step) stays correct.
        super().__init__(
            ft_is_angular=ft_is_angular,
            ft_names=ft_names,
            **kwargs,
        )

        # Replace the input projection so it accepts an extra is_motif channel.
        # Parent created Linear(n_inputs, hidden_size) with n_inputs = 6.
        hidden_size = self.config.hidden_size
        self.n_angle_features = len(ft_is_angular)
        self.n_aug_features = self.n_angle_features + 1  # +1 for is_motif
        self.inputs_to_hidden_dim = nn.Linear(self.n_aug_features, hidden_size)

        # SS embedding, additive at the input layer. Initialized small so the
        # untrained model is close to the base model's behavior.
        self.ss_embedding = nn.Embedding(ss_vocab_size, hidden_size)
        nn.init.normal_(self.ss_embedding.weight, mean=0.0, std=0.01)

        self.rama_lambda = float(rama_lambda)

        self._phi_idx = ft_names.index("phi")
        self._psi_idx = ft_names.index("psi")

        if pretrained_checkpoint is not None:
            self._load_pretrained_base_weights(pretrained_checkpoint)

        if freeze_pretrained:
            self._freeze_pretrained_weights()

        logging.info(
            f"BertForSSConditionedDiffusion: "
            f"n_aug_features={self.n_aug_features}, "
            f"n_angle_features={self.n_angle_features}, "
            f"rama_lambda={self.rama_lambda}, ss_vocab_size={ss_vocab_size}, "
            f"freeze_pretrained={freeze_pretrained}"
        )

    # ---------- forward ----------

    def forward(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor,
        ss_labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
          inputs:          (B, L, n_aug_features)  augmented angle+is_motif tensor
          timestep:        (B,) or (B, 1)
          attention_mask:  (B, L)
          ss_labels:       (B, L) int64 SS class indices, or None (no SS conditioning)
        Returns:
          predicted noise on the angle channels: (B, L, n_angle_features)
        """
        assert inputs.dim() == 3 and inputs.shape[-1] == self.n_aug_features, (
            f"Expected inputs of shape (B, L, {self.n_aug_features}), "
            f"got {tuple(inputs.shape)}"
        )
        batch_size, seq_length, _ = inputs.shape

        if position_ids is None:
            position_ids = (
                torch.arange(seq_length, device=inputs.device)
                .expand(batch_size, -1)
                .type_as(timestep)
            )

        ext_attn = attention_mask[:, None, None, :].type_as(attention_mask)
        ext_attn = (1.0 - ext_attn) * -10000.0

        h = self.inputs_to_hidden_dim(inputs)
        h = self.embeddings(h, position_ids=position_ids)

        if ss_labels is not None:
            h = h + self.ss_embedding(ss_labels)

        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1)
        h = h + time_encoded

        encoder_outputs = self.encoder(h, attention_mask=ext_attn, return_dict=True)
        sequence_output = encoder_outputs.last_hidden_state
        return self.token_decoder(sequence_output)

    def _load_pretrained_base_weights(self, checkpoint_path: str) -> None:
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "epoch_final.ckpt")

        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict(state.get("state_dict", state))

        # Expand the parent's Linear(n_angle_features, hidden) input projection
        # into our Linear(n_angle_features + 1, hidden): copy the pretrained
        # weights into the angle columns, zero the new is_motif column. This
        # makes the augmented model bit-identical to the base model at init
        # (given small ss_embedding init), so a frozen encoder gets the same
        # input distribution it was pretrained on.
        w_key = "inputs_to_hidden_dim.weight"
        b_key = "inputs_to_hidden_dim.bias"
        if w_key in state_dict:
            old_W = state_dict[w_key]                  # (hidden, n_angle_features)
            new_W = torch.zeros_like(self.inputs_to_hidden_dim.weight)
            if old_W.shape[1] != self.n_angle_features:
                raise ValueError(
                    f"Pretrained {w_key} has {old_W.shape[1]} input channels, "
                    f"expected {self.n_angle_features}."
                )
            new_W[:, : self.n_angle_features].copy_(old_W)
            # last column (is_motif) stays zero
            state_dict[w_key] = new_W
        if b_key in state_dict:
            state_dict[b_key] = state_dict[b_key].clone()

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # ss_embedding.* is expected to be missing (new module).
        logging.info(
            f"Loaded pretrained base weights from {checkpoint_path}. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )

    def _freeze_pretrained_weights(self) -> None:
        for name, param in self.named_parameters():
            if not (name.startswith("inputs_to_hidden_dim") or name.startswith("ss_embedding")):
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optimizer}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval

    # ---------- loss ----------

    def _get_loss_terms(  # type: ignore[override]
        self,
        batch,
        write_preds: Optional[str] = None,
    ):
        """
        Build the augmented input, run forward, mask motif positions out of
        the per-feature angular loss, and add the Ramachandran term.

        Returns a stacked tensor with one entry per angle feature. If
        rama_lambda > 0, an extra entry is appended for the prior term.

        Note: the parent BertForDiffusion training_step asserts
            len(loss_terms) == len(self.ft_names) (+1 if pairwise_dist_loss).
        Adding the Ramachandran term breaks that assertion, so we override
        training_step / validation_step below to log under our own scheme.
        """
        corrupted = batch["corrupted"]                         # (B, L, n_angle_features)
        motif_mask = batch["motif_mask"].to(corrupted.dtype)   # (B, L)
        motif_angles = batch["motif_angles"]                   # (B, L, n_angle_features)
        attn_mask = batch["attn_mask"]                         # (B, L)
        ss_labels = batch.get("ss_labels")                     # (B, L) or None

        # Replace corrupted angles with clean motif angles at motif positions.
        # This is the training-time analogue of the inpainting clamp at
        # sampling time. Teaches the model: motif tokens carry ground truth.
        mm = motif_mask.unsqueeze(-1)
        corrupted_with_motif = corrupted * (1.0 - mm) + motif_angles * mm

        # Append is_motif as a 7th input channel.
        aug_input = torch.cat([corrupted_with_motif, mm], dim=-1)

        predicted = self.forward(
            aug_input,
            batch["t"],
            attention_mask=attn_mask,
            ss_labels=ss_labels,
            position_ids=batch.get("position_ids"),
        )
        known_noise = batch["known_noise"]
        assert predicted.shape == known_noise.shape, (
            f"{predicted.shape} != {known_noise.shape}"
        )

        # valid = attention AND NOT motif: only score non-motif residues.
        valid = attn_mask.to(predicted.dtype) * (1.0 - motif_mask)
        valid_idx = torch.where(valid > 0.5)

        loss_terms = []
        for i in range(known_noise.shape[-1]):
            loss_fn = (
                self.loss_func[i] if isinstance(self.loss_func, list) else self.loss_func
            )
            spec = inspect.getfullargspec(loss_fn)
            kwargs = {}
            if "circle_penalty" in spec.args or "circle_penalty" in spec.kwonlyargs:
                kwargs["circle_penalty"] = self.circle_lambda
            l = loss_fn(
                predicted[valid_idx[0], valid_idx[1], i],
                known_noise[valid_idx[0], valid_idx[1], i],
                **kwargs,
            )
            loss_terms.append(l)

        # Optional Ramachandran prior on the one-step denoised angle estimate.
        # x0_hat = (x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar).
        if self.rama_lambda > 0.0 and ss_labels is not None:
            sqrt_alpha = batch["sqrt_alphas_cumprod_t"].view(-1, 1, 1)
            sqrt_one_minus_alpha = batch["sqrt_one_minus_alphas_cumprod_t"].view(-1, 1, 1)
            x0_hat = (corrupted_with_motif - sqrt_one_minus_alpha * predicted) / sqrt_alpha
            rama = ramachandran_loss(
                phi=x0_hat[..., self._phi_idx],
                psi=x0_hat[..., self._psi_idx],
                ss_labels=ss_labels,
                valid_mask=valid,
            )
            loss_terms.append(self.rama_lambda * rama)

        return torch.stack(loss_terms)

    # ---------- training / validation ----------
    # Override to handle the optional Ramachandran term cleanly.

    def _label_loss_terms(self, loss_terms: torch.Tensor) -> List[str]:
        names = list(self.ft_names)
        if self.rama_lambda > 0.0 and len(loss_terms) == len(names) + 1:
            names = names + ["rama"]
        return names

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)

        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            avg_loss = avg_loss + self.l1_lambda * l1_penalty

        names = self._label_loss_terms(loss_terms)
        loss_dict = {f"train_loss_{n}": v for n, v in zip(names, loss_terms)}
        loss_dict["train_loss"] = avg_loss
        self.log_dict(loss_dict)
        return avg_loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        with torch.no_grad():
            loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)

        names = self._label_loss_terms(loss_terms)
        loss_dict = {f"val_loss_{n}": self.all_gather(v) for n, v in zip(names, loss_terms)}
        loss_dict["val_loss"] = avg_loss
        self.log_dict(loss_dict, rank_zero_only=True)
        return {"val_loss": avg_loss}
