#!/usr/bin/env python3
"""
Batch driver for 2D histogram generation over a folder of CIF files.
Does not modify 2D_Histogram.py — loads it via importlib (non-importable module name).
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import traceback
from pathlib import Path


def _load_histogram_module(repo_root: Path):
    path = repo_root / "2D_Histogram.py"
    spec = importlib.util.spec_from_file_location("histogram_2d_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(
        description="Run 2D_Histogram.main() on every *.cif in a directory."
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=Path("."),
        help="Directory containing CIF files (non-recursive by default).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for *.cif recursively under --cif-dir.",
    )
    parser.add_argument(
        "--copy-to-npys",
        type=Path,
        default=None,
        metavar="DIR",
        help="After each structure, copy 2DOutput/<name>/Avg_Density.npy to DIR/<name>.npy for mof_2d_features.py.",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="min",
        help="Cutoff passed to 2D_Histogram.main (default: min).",
    )
    parser.add_argument(
        "--grid-distance",
        type=float,
        default=1.0,
        help="Grid spacing for histogram (default: 1).",
    )
    parser.add_argument(
        "--search-radius-ratio",
        type=float,
        default=2.5,
        help="SearchRadiusRatio for KD neighbor search (default: 2.5).",
    )
    parser.add_argument(
        "--distance-bin-step",
        type=float,
        default=1.0,
        help="Distance bin width in Angstrom for histogram distance axis (default: 1.0).",
    )
    parser.add_argument(
        "--double-distance-binning",
        action="store_true",
        help="Halve distance bin step (double distance-binning resolution).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib output per structure.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing Avg_Density.npy / Avg_all_xyz.npy under 2DOutput/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many CIFs (for testing).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    hmod = _load_histogram_module(repo_root)

    if args.recursive:
        cifs = sorted(args.cif_dir.rglob("*.cif"))
    else:
        cifs = sorted(args.cif_dir.glob("*.cif"))

    if args.limit is not None:
        cifs = cifs[: args.limit]

    if not cifs:
        print(f"No CIF files found under {args.cif_dir}", file=sys.stderr)
        sys.exit(1)

    if args.copy_to_npys is not None:
        args.copy_to_npys.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    plotting = not args.no_plot
    exportdata = not args.no_export

    for cif in cifs:
        cif = cif.resolve()
        stem = cif.stem
        try:
            hmod.main(
                cif=str(cif),
                cutoffdef=args.cutoff,
                grid_distance=args.grid_distance,
                SearchRadiusRatio=args.search_radius_ratio,
                distance_bin_step=args.distance_bin_step,
                double_distance_binning=args.double_distance_binning,
                BrutalChecking=False,
                plotting=plotting,
                exportdata=exportdata,
            )
            if args.copy_to_npys is not None and exportdata:
                src = repo_root / "2DOutput" / cif.name / "Avg_Density.npy"
                if not src.is_file():
                    failed.append((str(cif), f"missing export {src}"))
                    continue
                dst = args.copy_to_npys / f"{stem}.npy"
                shutil.copy2(src, dst)
            ok += 1
        except Exception as e:
            failed.append((str(cif), f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    print(f"Finished: {ok} succeeded, {len(failed)} failed (of {len(cifs)}).")
    for path, err in failed[:20]:
        print(f"  FAIL {path}\n{err}", file=sys.stderr)
    if len(failed) > 20:
        print(f"  ... and {len(failed) - 20} more failures.", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
