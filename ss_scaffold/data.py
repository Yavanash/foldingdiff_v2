import os
import time
import requests
import argparse
import logging
from pathlib import Path
from typing import Optional, List
from Bio import PDB
from Bio.PDB import PDBIO, Select

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

DEFAULT_MAX_STRUCTURES = 50
DEFAULT_MIN_HELIX_LEN  = 7
DEFAULT_MIN_HELIX_FRAC = 0.50
OUTPUT_DIR             = Path("../data/output_helices")
DOWNLOAD_DIR           = Path("../data/downloaded_pdbs")

HELIX_CLASS_NAMES = {
    1: "alpha",
    2: "omega",
    3: "pi",
    4: "gamma",
    5: "310",
    6: "left_alpha",
    7: "left_omega",
    8: "left_gamma",
    9: "2_7",
    10: "polyproline"
}


class HelixResidueSelect(Select):
    def __init__(self, chain_id, res_start, res_end):
        self.chain_id  = chain_id
        self.res_start = res_start
        self.res_end   = res_end

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        res_id = residue.get_id()[1]
        return self.res_start <= res_id <= self.res_end


def query_rcsb(max_results: int) -> List[str]:
    fetch_count = max_results * 3
    log.info(f"Querying RCSB for protein structures...")
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_composition",
                        "operator": "exact_match",
                        "value": "homomeric protein"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": fetch_count},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids  = [hit["identifier"] for hit in data.get("result_set", [])]
        log.info(f"  Found {len(ids)} candidate structures.")
        return ids
    except Exception as e:
        log.error(f"RCSB query failed: {e}")
        return []


def download_pdb(pdb_id: str, dest_dir: Path) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pdb_id}.pdb"
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_text(r.text)
        return dest
    except Exception as e:
        log.warning(f"  Could not download {pdb_id}: {e}")
        return None


def parse_helix_records(pdb_path: Path) -> List[dict]:
    helices = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HELIX "):
                continue
            try:
                chain_init  = line[19].strip()
                chain_end   = line[31].strip()
                res_start   = int(line[21:25].strip())
                res_end     = int(line[33:37].strip())
                helix_class = int(line[38:40].strip()) if line[38:40].strip() else 1
                helix_id    = line[11:14].strip()
                length      = res_end - res_start + 1
                if chain_init == chain_end and chain_init:
                    helices.append({
                        "chain":       chain_init,
                        "start":       res_start,
                        "end":         res_end,
                        "helix_id":    helix_id,
                        "helix_class": helix_class,
                        "length":      length
                    })
            except (ValueError, IndexError):
                continue
    return helices


def helix_fraction(pdb_path: Path) -> float:
    helix_res = set()
    total_res = set()
    helices   = parse_helix_records(pdb_path)
    for h in helices:
        for r in range(h["start"], h["end"] + 1):
            helix_res.add((h["chain"], r))
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    chain = line[21].strip()
                    res   = int(line[22:26].strip())
                    total_res.add((chain, res))
                except (ValueError, IndexError):
                    continue
    if not total_res:
        return 0.0
    return len(helix_res) / len(total_res)


def extract_helices(pdb_id: str, pdb_path: Path, output_dir: Path, min_length: int) -> int:
    helices = parse_helix_records(pdb_path)
    if not helices:
        log.info(f"  {pdb_id}: no HELIX records found.")
        return 0

    parser    = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))
    io        = PDBIO()
    io.set_structure(structure)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for h in helices:
        if h["length"] < min_length:
            continue
        class_name = HELIX_CLASS_NAMES.get(h["helix_class"], "helix")
        fname      = f"{pdb_id}_chain{h['chain']}_res{h['start']}-{h['end']}_{class_name}.pdb"
        out_path   = output_dir / fname
        try:
            io.save(str(out_path), HelixResidueSelect(h["chain"], h["start"], h["end"]))
            with open(out_path) as f:
                atom_lines = [l for l in f if l.startswith("ATOM")]
            if len(atom_lines) < 3:
                out_path.unlink(missing_ok=True)
                continue
            saved += 1
            log.info(f"  ✔ {fname}  ({h['length']} residues, {class_name}, {len(atom_lines)} atoms)")
        except Exception as e:
            log.warning(f"  Could not save {fname}: {e}")
            if out_path.exists():
                out_path.unlink()
    return saved


def write_manifest(output_dir: Path, stats: List[dict]):
    manifest = output_dir / "manifest.csv"
    with open(manifest, "w") as f:
        f.write("filename,pdb_id,chain,res_start,res_end,length,helix_class\n")
        for row in stats:
            f.write(
                f"{row['filename']},{row['pdb_id']},{row['chain']},"
                f"{row['res_start']},{row['res_end']},{row['length']},"
                f"{row['helix_class']}\n"
            )
    log.info(f"\n📄 Manifest written to {manifest}")


def run_pipeline(
    max_structures: int = DEFAULT_MAX_STRUCTURES,
    min_helix_len:  int = DEFAULT_MIN_HELIX_LEN,
    min_helix_frac: float = DEFAULT_MIN_HELIX_FRAC,
    output_dir:     Path = OUTPUT_DIR,
    download_dir:   Path = DOWNLOAD_DIR,
    pdb_ids:        Optional[List[str]] = None
):
    log.info("=" * 55)
    log.info("  Helix PDB Extraction Pipeline")
    log.info("=" * 55)

    if pdb_ids:
        ids = [p.upper() for p in pdb_ids]
        log.info(f"Using {len(ids)} user-provided PDB IDs.")
    else:
        ids = query_rcsb(max_structures)

    if not ids:
        log.error("No PDB IDs found. Exiting.")
        return

    total_helices = 0
    manifest_rows = []
    collected     = 0

    for i, pdb_id in enumerate(ids, 1):
        if not pdb_ids and collected >= max_structures:
            break

        log.info(f"\n[{i}/{len(ids)}] {pdb_id}")
        pdb_path = download_pdb(pdb_id, download_dir)
        if not pdb_path:
            continue

        time.sleep(0.1)

        frac = helix_fraction(pdb_path)
        if not pdb_ids and frac < min_helix_frac:
            log.info(f"  Skipping {pdb_id}: helix fraction {frac:.2f} < {min_helix_frac}")
            continue

        collected += 1
        raw_helices = parse_helix_records(pdb_path)
        saved       = extract_helices(pdb_id, pdb_path, output_dir, min_helix_len)
        total_helices += saved

        for h in raw_helices:
            if h["length"] < min_helix_len:
                continue
            class_name = HELIX_CLASS_NAMES.get(h["helix_class"], "helix")
            fname      = f"{pdb_id}_chain{h['chain']}_res{h['start']}-{h['end']}_{class_name}.pdb"
            if (output_dir / fname).exists():
                manifest_rows.append({
                    "filename":    fname,
                    "pdb_id":      pdb_id,
                    "chain":       h["chain"],
                    "res_start":   h["start"],
                    "res_end":     h["end"],
                    "length":      h["length"],
                    "helix_class": class_name
                })

    if manifest_rows:
        write_manifest(output_dir, manifest_rows)

    log.info("\n" + "=" * 55)
    log.info(f"  Done! Extracted {total_helices} helix segments")
    log.info(f"  Saved to: {output_dir.resolve()}")
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract helix-only PDB segments for model training.")
    parser.add_argument("--max-structures", type=int, default=DEFAULT_MAX_STRUCTURES)
    parser.add_argument("--min-helix-len",  type=int, default=DEFAULT_MIN_HELIX_LEN)
    parser.add_argument("--min-helix-frac", type=float, default=DEFAULT_MIN_HELIX_FRAC)
    parser.add_argument("--output-dir",     type=Path, default=OUTPUT_DIR)
    parser.add_argument("--download-dir",   type=Path, default=DOWNLOAD_DIR)
    parser.add_argument("--pdb-ids",        nargs="+", metavar="PDBID")
    args = parser.parse_args()

    run_pipeline(
        max_structures = args.max_structures,
        min_helix_len  = args.min_helix_len,
        min_helix_frac = args.min_helix_frac,
        output_dir     = args.output_dir,
        download_dir   = args.download_dir,
        pdb_ids        = args.pdb_ids
    )