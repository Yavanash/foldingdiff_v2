"""
RePaint-style motif inpainting at sampling time.

The base FoldingDiff reverse-diffusion loop (foldingdiff.sampling.p_sample_loop)
treats every residue identically. Here we wrap it: at every timestep, motif
residues are *clamped* to the forward-process noised version of their true
angles. The model never gets a chance to drift them.

Key formula (RePaint, Lugmayr et al. 2022):
    x_t[motif] = sqrt(alpha_bar_t) * x0[motif] + sqrt(1 - alpha_bar_t) * eps
where eps is fresh Gaussian noise (we sample once per timestep).

Reuses:
  - foldingdiff.beta_schedules.compute_alphas        (alpha schedule)
  - foldingdiff.utils.modulo_with_wrapped_range      (angle wrapping)
  - foldingdiff.modelling.BertForDiffusion structure (via the model arg)

The model is expected to be a BertForSSConditionedDiffusion: its forward signature
takes (inputs[B,L,n_aug], timestep, attention_mask, ss_labels, position_ids).
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
from torch import nn
from tqdm.auto import tqdm

from foldingdiff import beta_schedules, utils


@torch.no_grad()
def _q_sample_motif(
    x0: torch.Tensor,
    t_index: int,
    alpha_terms: dict,
    angular_idx: Sequence[int],
) -> torch.Tensor:
    """
    Forward-process q_sample applied only to the motif slice.

    x0:           (B, L_motif, n_features) clean motif angles
    t_index:      scalar int timestep
    alpha_terms:  dict from beta_schedules.compute_alphas
    angular_idx:  indices of angular features (so we wrap them after noising)
    """
    sqrt_alpha = alpha_terms["sqrt_alphas_cumprod"][t_index]
    sqrt_one_minus = alpha_terms["sqrt_one_minus_alphas_cumprod"][t_index]
    eps = torch.randn_like(x0)
    x_t = sqrt_alpha * x0 + sqrt_one_minus * eps
    if angular_idx:
        x_t[..., list(angular_idx)] = utils.modulo_with_wrapped_range(
            x_t[..., list(angular_idx)], range_min=-torch.pi, range_max=torch.pi
        )
    return x_t


@torch.no_grad()
def p_sample_loop_with_motif(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    ss_labels: torch.Tensor,
    motif_mask: torch.Tensor,
    motif_angles: torch.Tensor,
    is_angle: Sequence[bool],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Reverse-diffusion sampling with motif inpainting.

    Args:
      model:          BertForSSConditionedDiffusion-compatible model.
      lengths:        per-example unpadded sequence lengths.
      noise:          (B, L, n_angle_features) initial Gaussian noise.
      timesteps:      total diffusion steps T.
      betas:          (T,) beta schedule.
      ss_labels:      (B, L) int64 SS class indices.
      motif_mask:     (B, L) {0, 1} motif indicator.
      motif_angles:   (B, L, n_angle_features) clean motif angles (zeros at
                      non-motif positions; values are read only where motif_mask==1).
      is_angle:       per-feature angular flags (matches noise.shape[-1]).
      disable_pbar:   silence tqdm.

    Returns:
      (T, B, L, n_angle_features) tensor of x_t at every step (CPU).
    """
    device = next(model.parameters()).device
    img = noise.to(device)
    B = img.shape[0]
    n_angle_features = img.shape[-1]
    assert motif_mask.shape == ss_labels.shape == img.shape[:2]
    assert motif_angles.shape == img.shape

    ss_labels = ss_labels.to(device)
    motif_mask = motif_mask.to(device)
    motif_angles = motif_angles.to(device)

    alpha_terms = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_terms["alphas"])
    angular_idx = [i for i, a in enumerate(is_angle) if a]

    # Precompute per-example attention mask from `lengths`.
    attn_mask = torch.zeros((B, img.shape[1]), device=device)
    for i, l in enumerate(lengths):
        attn_mask[i, :l] = 1.0

    # Initial motif clamp at t = T (heaviest noise level).
    if motif_mask.any():
        mm = motif_mask.unsqueeze(-1)
        clamp = _q_sample_motif(motif_angles, timesteps - 1, alpha_terms, angular_idx)
        img = img * (1.0 - mm) + clamp * mm

    imgs = []
    for t_idx in tqdm(
        reversed(range(timesteps)),
        total=timesteps,
        desc="ss-scaffold sampling",
        disable=disable_pbar,
    ):
        # Build the augmented input the model expects: angles + is_motif channel.
        mm = motif_mask.unsqueeze(-1).to(img.dtype)
        aug_input = torch.cat([img, mm], dim=-1)

        t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps_pred = model(
            aug_input,
            t_vec,
            attention_mask=attn_mask,
            ss_labels=ss_labels,
        )

        beta_t = betas[t_idx]
        sqrt_one_minus_alpha_t = alpha_terms["sqrt_one_minus_alphas_cumprod"][t_idx]
        sqrt_recip_alpha_t = sqrt_recip_alphas[t_idx]

        # Reverse-process mean (DDPM eq. 11).
        model_mean = sqrt_recip_alpha_t * (
            img - beta_t * eps_pred / sqrt_one_minus_alpha_t
        )

        if t_idx == 0:
            img = model_mean
        else:
            posterior_var_t = alpha_terms["posterior_variance"][t_idx]
            z = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_var_t) * z

        # Wrap angular features.
        for j in angular_idx:
            img[..., j] = utils.modulo_with_wrapped_range(
                img[..., j], range_min=-torch.pi, range_max=torch.pi
            )

        # RePaint clamp: at every step, overwrite motif positions with a fresh
        # forward-noise of the clean motif at the *current* noise level.
        if motif_mask.any() and t_idx > 0:
            mm = motif_mask.unsqueeze(-1)
            clamp = _q_sample_motif(motif_angles, t_idx - 1, alpha_terms, angular_idx)
            img = img * (1.0 - mm) + clamp * mm
        elif motif_mask.any() and t_idx == 0:
            # Final step: hard-set motif to ground truth (no noise).
            mm = motif_mask.unsqueeze(-1)
            img = img * (1.0 - mm) + motif_angles * mm

        imgs.append(img.detach().cpu())

    return torch.stack(imgs)
