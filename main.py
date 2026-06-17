#!/usr/bin/env python3
"""
main.py
-------
CLI entry point to execute our modular `voxelizer` package.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from voxelizer import stl_to_voxels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an STL file to a 3-D voxel grid array via modular pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stl_file", help="Input STL file path.")
    parser.add_argument(
        "--resolution", "-r", type=int, default=64, metavar="N",
        help="Voxels along the longest axis. Ignored when --voxel-size is set."
    )
    parser.add_argument(
        "--voxel-size", "-v", type=float, default=None, metavar="SIZE",
        help="Physical voxel edge length in mesh units. Overrides --resolution."
    )
    parser.add_argument(
        "--subsample", "-s", type=int, default=4, metavar="N",
        help="Sub-samples per axis per voxel. 4 -> 64 samples/voxel."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, metavar="FILE",
        help="Output .npy file. Defaults to <stl_name>.npy."
    )
    parser.add_argument(
        "--batch-size", type=int, default=500_000, metavar="N",
        help="Points per inside/outside batch processing step."
    )

    args = parser.parse_args()

    stl_path = Path(args.stl_file)
    if not stl_path.exists():
        sys.exit(f"Error: file not found -> {stl_path}")

    out_path = Path(args.output) if args.output else stl_path.with_suffix(".npy")

    # Call our package pipeline
    try:
        grid = stl_to_voxels(
            stl_path=stl_path,
            resolution=args.resolution,
            voxel_size=args.voxel_size,
            subsample=args.subsample,
            batch_size=args.batch_size,
        )
    except Exception as e:
        sys.exit(f"Pipeline Execution Failed: {e}")

    # Save outputs
    np.save(out_path, grid)
    print(f"\nSaved    {out_path}")
    print(f"Shape    {grid.shape}   dtype={grid.dtype}")
    print(f"Values   min={grid.min():.4f}  max={grid.max():.4f}  mean={grid.mean():.4f}")


if __name__ == "__main__":
    main()