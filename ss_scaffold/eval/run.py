"""End-to-end eval: generated PDBs -> MPNN -> ESMFold -> metrics table.

Pipeline per generated backbone:
    1. ProteinMPNN samples N sequences (CA-only).
    2. ESMFold predicts a structure for each sequence (records mean pLDDT).
    3. For each (gen, pred) pair: scRMSD, scTM, optional motif-RMSD, DSSP fidelity.
    4. Aggregate per-backbone: take min scRMSD across sequences, paired pLDDT,
       and mark designable per Yim et al. (scRMSD < 2 Å AND pLDDT > 70).
    5. Across designable set: pairwise-TM diversity + cluster count.

Outputs:
    <out>/per_sequence.csv   one row per (backbone, sequence)
    <out>/per_backbone.csv   one row per backbone with min/best aggregates
    <out>/summary.json       headline numbers (designability rate, diversity, ...)
    <out>/seqs/              fasta files of sampled sequences
    <out>/folded/            ESMFold-predicted PDBs

Usage:
    python -m ss_scaffold.eval.run \\
        --generated-dir generated/ \\
        --out eval_out/ \\
        --metadata generated/metadata.json \\
        --n-seqs 8

The --metadata flag points at the file ss_scaffold/sample.py writes; if present
the motif range and SS string are read from it for motif-RMSD + DSSP fidelity.
"""
from __future__ import annotations

import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ss_scaffold.eval import diversity, esmfold, metrics, mpnn


def _build_ss_string(L: int, motif_start: int, motif_end: int, motif_class: str) -> str:
    chars = ["C"] * L
    for i in range(motif_start, motif_end):
        chars[i] = motif_class
    return "".join(chars)


def _load_metadata(path: Optional[str]):
    if not path or not Path(path).is_file():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generated-dir", required=True,
                   help="Dir of *.pdb produced by ss_scaffold.sample")
    p.add_argument("--out", required=True)
    p.add_argument("--metadata", default=None,
                   help="Path to metadata.json from sampling (for motif/SS info)")
    p.add_argument("--n-seqs", type=int, default=8)
    p.add_argument("--mpnn-temperature", type=float, default=0.1)
    p.add_argument("--mpnn-script", default=None)
    p.add_argument("--esm-chunk-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--rmsd-threshold", type=float, default=2.0)
    p.add_argument("--plddt-threshold", type=float, default=70.0)
    p.add_argument("--diversity-tm-threshold", type=float, default=0.5)
    p.add_argument("--skip-diversity", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    (out / "seqs").mkdir(parents=True, exist_ok=True)
    (out / "folded").mkdir(parents=True, exist_ok=True)

    gen_pdbs = sorted(glob(str(Path(args.generated_dir) / "*.pdb")))
    if not gen_pdbs:
        raise SystemExit(f"No PDBs in {args.generated_dir}")
    logging.info(f"Evaluating {len(gen_pdbs)} generated backbones")

    meta = _load_metadata(args.metadata)
    motif_residues = None
    requested_ss = None
    if meta is not None:
        ms, me = meta["motif_target"]
        motif_residues = list(range(ms, me))
        requested_ss = _build_ss_string(meta["total_length"], ms, me, meta["motif_class"])
        logging.info(f"Motif residues {ms}-{me}; SS class {meta['motif_class']}")

    folder = esmfold.ESMFoldRunner(device=args.device, chunk_size=args.esm_chunk_size)

    per_seq_rows = []
    per_bb_rows = []

    for gen_pdb in gen_pdbs:
        bname = Path(gen_pdb).stem
        logging.info(f"=== {bname} ===")

        try:
            seqs = mpnn.design_sequences(
                gen_pdb,
                n_sequences=args.n_seqs,
                temperature=args.mpnn_temperature,
                mpnn_script=args.mpnn_script,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(f"MPNN failed on {bname}: {e}")
            continue

        # Persist fasta for reproducibility.
        with open(out / "seqs" / f"{bname}.fasta", "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">{bname}_seq{i}\n{s}\n")

        seq_dir = out / "folded" / bname
        fold_results = esmfold.fold_many(seqs, str(seq_dir), runner=folder, prefix="pred")

        sc_rmsds, plddts, sc_tms, mot_rmsds, dssp_accs = [], [], [], [], []
        for i, ((pred_pdb, plddt), seq) in enumerate(zip(fold_results, seqs)):
            row = {"backbone": bname, "seq_idx": i, "sequence": seq}
            if pred_pdb is None:
                row.update({"sc_rmsd": np.nan, "sc_tm": np.nan,
                            "mean_plddt": np.nan, "motif_rmsd": np.nan, "dssp_acc": np.nan})
                per_seq_rows.append(row)
                continue
            try:
                rmsd = metrics.sc_rmsd(gen_pdb, pred_pdb)
                tm = metrics.sc_tm(gen_pdb, pred_pdb)
                mp_lddt = float(np.mean(plddt))
                mot = metrics.motif_rmsd(gen_pdb, pred_pdb, motif_residues) \
                    if motif_residues is not None else np.nan
                dssp_acc = metrics.dssp_fidelity(pred_pdb, requested_ss)[0] \
                    if requested_ss is not None else np.nan
            except Exception as e:  # noqa: BLE001
                logging.warning(f"  metric failure on seq {i}: {e}")
                rmsd = tm = mp_lddt = mot = dssp_acc = np.nan

            row.update({"sc_rmsd": rmsd, "sc_tm": tm, "mean_plddt": mp_lddt,
                        "motif_rmsd": mot, "dssp_acc": dssp_acc,
                        "predicted_pdb": pred_pdb})
            per_seq_rows.append(row)
            sc_rmsds.append(rmsd); plddts.append(mp_lddt); sc_tms.append(tm)
            mot_rmsds.append(mot); dssp_accs.append(dssp_acc)

        if not sc_rmsds:
            continue
        # Best sequence = min scRMSD.
        best = int(np.nanargmin(sc_rmsds))
        per_bb_rows.append({
            "backbone": bname,
            "min_sc_rmsd": float(sc_rmsds[best]),
            "paired_plddt": float(plddts[best]),
            "best_sc_tm": float(sc_tms[best]),
            "best_motif_rmsd": float(mot_rmsds[best]) if mot_rmsds else np.nan,
            "best_dssp_acc": float(dssp_accs[best]) if dssp_accs else np.nan,
            "designable": metrics.is_designable(
                sc_rmsds[best], plddts[best],
                args.rmsd_threshold, args.plddt_threshold,
            ),
            "best_predicted_pdb": str(seq_dir / f"pred_{best:03d}.pdb"),
        })

    per_seq_df = pd.DataFrame(per_seq_rows)
    per_bb_df = pd.DataFrame(per_bb_rows)
    per_seq_df.to_csv(out / "per_sequence.csv", index=False)
    per_bb_df.to_csv(out / "per_backbone.csv", index=False)

    summary = {
        "n_backbones": int(len(per_bb_df)),
        "designability_rate": float(per_bb_df["designable"].mean()) if len(per_bb_df) else 0.0,
        "median_min_sc_rmsd": float(per_bb_df["min_sc_rmsd"].median()) if len(per_bb_df) else np.nan,
        "median_paired_plddt": float(per_bb_df["paired_plddt"].median()) if len(per_bb_df) else np.nan,
    }
    if requested_ss is not None and len(per_bb_df):
        summary["median_best_dssp_acc"] = float(per_bb_df["best_dssp_acc"].median())
        summary["median_best_motif_rmsd"] = float(per_bb_df["best_motif_rmsd"].median())

    if not args.skip_diversity and len(per_bb_df):
        designable = per_bb_df[per_bb_df["designable"]]
        if len(designable) >= 2:
            pdbs = designable["best_predicted_pdb"].tolist()
            tm_mat = diversity.pairwise_tm(pdbs)
            np.save(out / "designable_pairwise_tm.npy", tm_mat)
            iu = np.triu_indices_from(tm_mat, k=1)
            summary["designable_n"] = int(len(designable))
            summary["mean_pairwise_tm_designable"] = float(np.nanmean(tm_mat[iu]))
            summary["cluster_count_designable"] = int(
                diversity.cluster_count(tm_mat, threshold=args.diversity_tm_threshold)
            )

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote {out}/summary.json")
    logging.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
