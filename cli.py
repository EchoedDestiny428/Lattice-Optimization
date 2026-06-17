"""
cli.py — Command-line interface for stl_voxelizer.

Usage
-----
python -m stl_voxelizer model.stl [options]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from .voxelize import stl_to_voxels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m stl_voxelizer",
        description="Convert an STL file to a 3-D voxel grid (float32, values 0–1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "stl_file",
        help="Input STL file path.",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int, default=64, metavar="N",
        help="Voxels along the longest axis. Ignored when --voxel-size is set.",
    )
    parser.add_argument(
        "--voxel-size", "-v",
        type=float, default=None, metavar="SIZE",
        help="Physical voxel edge length in mesh units. Overrides --resolution.",
    )
    parser.add_argument(
        "--subsample", "-s",
        type=int, default=4, metavar="N",
        help="Sub-samples per axis per voxel. 4 → 64 pts/voxel. 8 → 512 pts/voxel.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None, metavar="FILE",
        help="Output .npy path. Defaults to <input_name>.npy in the same folder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int, default=500_000, metavar="N",
        help="Points per inside/outside batch. Lower if you hit memory limits.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    stl_path = Path(args.stl_file)
    if not stl_path.exists():
        sys.exit(f"Error: file not found → {stl_path}")

    out_path = Path(args.output) if args.output else stl_path.with_suffix(".npy")

    if not args.quiet:
        print(f"Input    {stl_path}")

    grid = stl_to_voxels(
        stl_path   = stl_path,
        resolution = args.resolution,
        voxel_size = args.voxel_size,
        subsample  = args.subsample,
        batch_size = args.batch_size,
        verbose    = not args.quiet,
    )

    np.save(out_path, grid)

    if not args.quiet:
        print(f"\nSaved    {out_path}")
        print(f"Shape    {grid.shape}   dtype={grid.dtype}")
        print(f"Values   min={grid.min():.4f}  max={grid.max():.4f}  mean={grid.mean():.4f}")
        print(f"Fill     {grid.mean() * 100:.1f}% of bounding box occupied")