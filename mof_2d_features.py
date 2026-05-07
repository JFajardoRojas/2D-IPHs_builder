#!/usr/bin/env python3
"""
Build a single, ordered per-MOF table from flattened 2D-histogram .npy files.

One row per MOF; columns are MOF_name (or custom id column) and feature_0..feature_{n-1}
in fixed order (Fortran-flattened vector, matching the saved Avg_Density arrays).

Writes exactly one CSV (all features). Pruning and train/val/test splits are handled by
prune_feature_columns.py. This module does not read label files — use
attach_mof_features_to_labels.py to merge with labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def normalize_mof_key(name: str, strip_chg: bool = True) -> str:
    s = str(name).strip()
    if s.lower().endswith(".cif"):
        s = s[:-4]
    if strip_chg and s.endswith("_CHG"):
        s = s[: -len("_CHG")]
    return s


def load_histogram_npys_flat(folder: Path, strip_chg: bool = True) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for path in sorted(folder.glob("*.npy")):
        key = normalize_mof_key(path.stem, strip_chg=strip_chg)
        arr = np.load(path)
        out[key] = np.asarray(arr).ravel(order="F")
    return out


def load_histogram_2doutput(root: Path, strip_chg: bool = True) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        npy = sub / "Avg_Density.npy"
        if not npy.is_file():
            continue
        key = normalize_mof_key(sub.name, strip_chg=strip_chg)
        arr = np.load(npy)
        out[key] = np.asarray(arr).ravel(order="F")
    return out


def histogram_dict_to_dataframe(
    hist: Mapping[str, np.ndarray],
    mof_column: str = "MOF_name",
    feature_prefix: str = "feature_",
) -> pd.DataFrame:
    """
    Rows sorted by MOF key; feature columns sorted by index (feature_0, feature_1, ...).
    """
    if not hist:
        return pd.DataFrame(columns=[mof_column])

    lengths = {k: int(v.size) for k, v in hist.items()}
    n = lengths[next(iter(hist))]
    bad = {k: L for k, L in lengths.items() if L != n}
    if bad:
        sample = next(iter(bad.items()))
        raise ValueError(f"Inconsistent feature lengths (expected {n}): e.g. {sample[0]!r} has {sample[1]}")

    feat_cols = [f"{feature_prefix}{i}" for i in range(n)]
    keys = sorted(hist.keys())
    matrix = np.stack([hist[k] for k in keys], axis=0)
    df = pd.DataFrame(matrix, columns=feat_cols)
    df.insert(0, mof_column, keys)
    return df


def build_from_npys(
    npy_folder: Path,
    strip_chg: bool = True,
    mof_column: str = "MOF_name",
    feature_prefix: str = "feature_",
) -> pd.DataFrame:
    hist = load_histogram_npys_flat(npy_folder, strip_chg=strip_chg)
    return histogram_dict_to_dataframe(hist, mof_column=mof_column, feature_prefix=feature_prefix)


def build_from_2doutput(
    twod_root: Path,
    strip_chg: bool = True,
    mof_column: str = "MOF_name",
    feature_prefix: str = "feature_",
) -> pd.DataFrame:
    hist = load_histogram_2doutput(twod_root, strip_chg=strip_chg)
    return histogram_dict_to_dataframe(hist, mof_column=mof_column, feature_prefix=feature_prefix)


def save_feature_table(
    df: pd.DataFrame,
    out_csv: Path,
    manifest_path: Optional[Path] = None,
    mof_column: str = "MOF_name",
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if manifest_path is not None:
        feat_cols = [c for c in df.columns if c != mof_column]
        manifest = {
            "mof_column": mof_column,
            "n_mofs": int(len(df)),
            "n_features": len(feat_cols),
            "feature_columns": feat_cols,
            "output_csv": str(out_csv.resolve()),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-MOF flattened 2D-histogram feature table from .npy files.")
    parser.add_argument("--histogram-mode", choices=("npys", "2doutput"), default="npys")
    parser.add_argument("--npy-folder", type=Path, default=Path("npys"))
    parser.add_argument("--2doutput-root", dest="twod_output_root", type=Path, default=Path("2DOutput"))
    parser.add_argument("--output", type=Path, default=Path("MOF_2D_features.csv"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON sidecar with column metadata.")
    parser.add_argument("--mof-column", type=str, default="MOF_name")
    parser.add_argument("--feature-prefix", type=str, default="feature_")
    parser.add_argument("--keep-chg-suffix", action="store_true", help="Do not strip trailing _CHG from stems.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    strip_chg = not args.keep_chg_suffix

    if args.histogram_mode == "npys":
        folder = args.npy_folder if args.npy_folder.is_absolute() else root / args.npy_folder
        df = build_from_npys(folder, strip_chg=strip_chg, mof_column=args.mof_column, feature_prefix=args.feature_prefix)
    else:
        twod = args.twod_output_root if args.twod_output_root.is_absolute() else root / args.twod_output_root
        df = build_from_2doutput(twod, strip_chg=strip_chg, mof_column=args.mof_column, feature_prefix=args.feature_prefix)

    if df.empty or (len(df.columns) == 1 and args.mof_column in df.columns):
        print("No histogram vectors loaded — check folder path.", file=sys.stderr)
        sys.exit(1)

    n_feat = len(df.columns) - 1

    out = args.output if args.output.is_absolute() else root / args.output
    man = args.manifest
    if man is not None and not man.is_absolute():
        man = root / man
    save_feature_table(df, out, manifest_path=man, mof_column=args.mof_column)
    print(f"Wrote {out}  ({len(df)} MOFs × {n_feat} features)")


if __name__ == "__main__":
    main()
