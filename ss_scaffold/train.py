"""
Minimal training entrypoint for ss_scaffold.

Reuses foldingdiff's dataset + Lightning training loop machinery via the parent
class. The only ss_scaffold-specific wiring is:
  - Wrap the noised dataset in SSAnnotatedAnglesDataset
  - Instantiate BertForSSConditionedDiffusion instead of BertForDiffusion

Run:
  python -m ss_scaffold.train --pdb-dir data/cath_s40 --out runs/ss_scaffold_v1

This is a starting-point script, not a polished CLI. Tune for your hardware.
"""
from __future__ import annotations

import argparse
import logging
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig

from foldingdiff.datasets import (
    CathCanonicalAnglesOnlyDataset,
    NoisedAnglesDataset,
)

from ss_scaffold.dataset import SSAnnotatedAnglesDataset
from ss_scaffold.model import BertForSSConditionedDiffusion


def build_datasets(args):
    base_train = CathCanonicalAnglesOnlyDataset(
        pdbs=args.pdb_dir,
        split="train",
        pad=args.pad,
        trim_strategy=args.trim_strategy,
        min_length=0,  # Disable min_length filter for short sequences
    )
    base_val = CathCanonicalAnglesOnlyDataset(
        pdbs=args.pdb_dir,
        split="validation",
        pad=args.pad,
        trim_strategy=args.trim_strategy,
        zero_center=True,
        min_length=0,  # Disable min_length filter for short sequences
    )
    base_val.set_masked_means(base_train.get_masked_means())

    noised_train = NoisedAnglesDataset(
        dset=base_train,
        dset_key="angles",
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        nonangular_variance=1.0,
        angular_variance=args.angular_variance,
    )
    noised_val = NoisedAnglesDataset(
        dset=base_val,
        dset_key="angles",
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        nonangular_variance=1.0,
        angular_variance=args.angular_variance,
    )

    train_dset = SSAnnotatedAnglesDataset(
        wrapped=noised_train,
        pdb_dir=args.pdb_dir,
        cache_dir=args.cache_dir,
        motif_mode=args.motif_mode,
        motif_min_len=args.motif_min_len,
        motif_max_len=args.motif_max_len,
        p_no_motif=args.p_no_motif,
    )
    val_dset = SSAnnotatedAnglesDataset(
        wrapped=noised_val,
        pdb_dir=args.pdb_dir,
        cache_dir=args.cache_dir,
        motif_mode=args.motif_mode,
        motif_min_len=args.motif_min_len,
        motif_max_len=args.motif_max_len,
        p_no_motif=args.p_no_motif,
    )
    return train_dset, val_dset, base_train.get_masked_means()


def build_model(args, steps_per_epoch: int) -> BertForSSConditionedDiffusion:
    config = BertConfig(
        max_position_embeddings=args.pad,
        num_attention_heads=args.num_heads,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size if args.intermediate_size else args.hidden_size * 4,
        num_hidden_layers=args.num_layers,
        position_embedding_type=args.position_embedding_type,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        use_cache=False,
    )

    return BertForSSConditionedDiffusion(
        config=config,
        rama_lambda=args.rama_lambda,
        time_encoding=args.time_encoding,
        decoder=args.decoder,
        lr=args.lr,
        loss=args.loss,
        l2=args.l2,
        l1=args.l1,
        circle_reg=args.circle_reg,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        lr_scheduler=args.lr_scheduler,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_pretrained=args.freeze_pretrained,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--pad", type=int, default=128)
    p.add_argument("--trim-strategy", default="randomcrop")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-schedule", default="cosine")
    p.add_argument("--angular-variance", type=float, default=1.0)
    p.add_argument("--p-no-motif", type=float, default=0.3)
    p.add_argument("--motif-mode", default="mixed",
                   choices=["ss_run", "arbitrary_span", "mixed"],
                   help="ss_run = legacy single-class motif; arbitrary_span = "
                        "any contiguous chunk (matches inference on arbitrary "
                        "input PDBs); mixed = 50/50 of both.")
    p.add_argument("--motif-min-len", type=int, default=6)
    p.add_argument("--motif-max-len", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--position-embedding-type", default="relative_key",
                   choices=["absolute", "relative_key", "relative_key_query"])
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rama-lambda", type=float, default=0.1)
    p.add_argument("--time-encoding", default="gaussian_fourier")
    p.add_argument("--decoder", default="mlp")
    p.add_argument("--loss", default="smooth_l1")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-scheduler", default="LinearWarmup")
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--circle-reg", type=float, default=0.0)
    p.add_argument("--pretrained-checkpoint", default=None)
    p.add_argument("--freeze-pretrained", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dset, val_dset, training_means = build_datasets(args)
    if training_means is not None:
        np.save(out_dir / "training_means.npy", training_means)
        logging.info(f"Saved training_means.npy with shape {training_means.shape}: {training_means}")
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    steps_per_epoch = max(1, len(train_loader))
    model = build_model(args, steps_per_epoch)

    # Write artifacts that from_dir (used by sample.py) needs.
    model.config.save_pretrained(out_dir)
    training_args = {
        "angles_definitions": "canonical-full-angles",
        "time_encoding": args.time_encoding,
        "decoder": args.decoder,
        "rama_lambda": args.rama_lambda,
        "ft_names": ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
    }
    with open(out_dir / "training_args.json", "w") as f:
        json.dump(training_args, f, indent=2)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        default_root_dir=str(out_dir),
        callbacks=callbacks,
        log_every_n_steps=20,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
