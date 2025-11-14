#!/usr/bin/env python3
"""
Build snapshot matrices (u', v', w', cp') from a series of RHEA .h5 files.

Features
--------
- Reads u, v, w, c_p from 3D channel / duct DNS.
- Removes one ghost cell layer in each direction.
- Computes and subtracts the appropriate mean:
  * channel: homogeneous in x,z => < · >_{x,z} = f(y)
  * duct   : homogeneous in x   => < · >_{x}   = f(y,z)
- Stores:
  * u, v, w, uvw : fluctuations flattened as (space, snapshots)
  * cp_prime     : cp' flattened as (space, snapshots)
  * cp_mean_yz   : mean cp field as (ny, nz, snapshots)
  * y, z         : 1D coordinates (no ghosts)
  * nx, ny, nz   : grid sizes

Usage example
-------------
python build_snapshots.py \\
  --case-num 80 \\
  --data-dir /path/to/h5_files \\
  --output-dir /path/to/output \\
  --start-file 3d_turbulent_channel_flow_7500000.h5 \\
  --end-file   3d_turbulent_channel_flow_8400000.h5 \\
  --geometry channel
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def list_snapshots(
    data_dir: Path, start_file: str, end_file: str
) -> List[str]:
    """
    Return sorted list of .h5 snapshots between start_file and end_file (inclusive),
    assuming lexicographical order is consistent with time/iteration.
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    selected = [f for f in all_files if start_file <= f <= end_file]
    selected.sort()
    if not selected:
        raise RuntimeError(
            f"No .h5 files found in {data_dir} between {start_file} and {end_file}"
        )
    print(f"Found {len(selected)} snapshots in range [{start_file}, {end_file}].")
    return selected


def parse_iteration(filename: str) -> int:
    """
    Extract integer iteration from a name like '..._7500000.h5'
    by taking the last underscore-separated token and stripping extension.
    """
    last_part = filename.split("_")[-1]  # '7500000.h5'
    base, _ = os.path.splitext(last_part)
    return int(base)


def load_fields(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray, np.ndarray]:
    """
    Load u, v, w, c_p, y, z from a .h5 file, transpose to (x, y, z) and remove
    one ghost layer in each direction.

    Returns:
        u, v, w, cp, y_1d, z_1d
        - u, v, w, cp: arrays of shape (nx, ny, nz) without ghosts
        - y_1d: array (ny,) taking the line y[x=0, z=0]
        - z_1d: array (nz,) taking the line z[x=0, y=0]
    """
    with h5py.File(path, "r") as f:
        u = np.transpose(f["u"][...], (2, 1, 0))
        v = np.transpose(f["v"][...], (2, 1, 0))
        w = np.transpose(f["w"][...], (2, 1, 0))
        cp = np.transpose(f["c_p"][...], (2, 1, 0))
        y = np.transpose(f["y"][...], (2, 1, 0))
        z = np.transpose(f["z"][...], (2, 1, 0))

    # Remove ghost cells: (nx, ny, nz) -> (nx-2, ny-2, nz-2)
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))
    u = u[sl]
    v = v[sl]
    w = w[sl]
    cp = cp[sl]
    y = y[sl]
    z = z[sl]

    # 1D coordinates (assuming structured channel/duct)
    y_1d = y[0, :, 0].copy()
    z_1d = z[0, 0, :].copy()

    return u, v, w, cp, y_1d, z_1d


def subtract_means_inplace(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cp: np.ndarray,
    geometry: str,
) -> np.ndarray:
    """
    Subtract appropriate mean IN PLACE for the given geometry.

    Args:
        u, v, w, cp: arrays of shape (nx, ny, nz) (no ghosts)
        geometry: 'channel' or 'duct'

    Returns:
        cp_mean_yz: array of shape (ny, nz) with mean cp field:
            - channel: <cp>_{x,z}(y) expanded uniformly in z
            - duct   : <cp>_{x}(y,z)
    """
    nx, ny, nz = u.shape

    if geometry == "channel":
        # hom. in x,z => mean depends only on y
        u_mean_y = u.mean(axis=(0, 2))   # (ny,)
        v_mean_y = v.mean(axis=(0, 2))
        w_mean_y = w.mean(axis=(0, 2))
        cp_mean_y = cp.mean(axis=(0, 2))

        # expand to (ny, nz) for uniform interface
        cp_mean_yz = np.broadcast_to(cp_mean_y[:, None], (ny, nz))

        # subtract in place using broadcasting
        u -= u_mean_y[np.newaxis, :, np.newaxis]
        v -= v_mean_y[np.newaxis, :, np.newaxis]
        w -= w_mean_y[np.newaxis, :, np.newaxis]
        cp -= cp_mean_y[np.newaxis, :, np.newaxis]

    elif geometry == "duct":
        # hom. in x => mean depends on y,z
        u_mean_yz = u.mean(axis=0)       # (ny, nz)
        v_mean_yz = v.mean(axis=0)
        w_mean_yz = w.mean(axis=0)
        cp_mean_yz = cp.mean(axis=0)

        cp_mean_yz = cp_mean_yz  # (ny, nz)

        # subtract in place
        u -= u_mean_yz[None, :, :]
        v -= v_mean_yz[None, :, :]
        w -= w_mean_yz[None, :, :]
        cp -= cp_mean_yz[None, :, :]

    else:
        raise ValueError(f"Unknown geometry: {geometry!r}. Use 'channel' or 'duct'.")

    return cp_mean_yz


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def build_snapshots(
    case_num: int,
    data_dir: Path,
    output_dir: Path,
    start_file: str,
    end_file: str,
    geometry: str,
) -> None:
    """Main routine to build snapshot matrices and save them to NPZ."""
    t0 = time.time()

    snapshots = list_snapshots(data_dir, start_file, end_file)
    m = len(snapshots)

    # Use first snapshot to determine shapes and allocate matrices
    first_path = data_dir / snapshots[0]
    u0, v0, w0, cp0, y_1d, z_1d = load_fields(first_path)
    nx, ny, nz = u0.shape
    n = nx * ny * nz

    print(f"Grid (no ghosts): nx={nx}, ny={ny}, nz={nz}  --> n={n} points")
    print(f"Allocating snapshot matrices for {m} snapshots...")

    u_prime = np.empty((n, m), dtype=np.float64)
    v_prime = np.empty((n, m), dtype=np.float64)
    w_prime = np.empty((n, m), dtype=np.float64)
    uvw_prime = np.empty((3 * n, m), dtype=np.float64)

    cp_prime = np.empty((n, m), dtype=np.float64)
    cp_mean_yz = np.empty((ny, nz, m), dtype=np.float64)

    # Process first snapshot
    print(f"[1/{m}] Processing {snapshots[0]}")
    cp_mean_yz_0 = subtract_means_inplace(u0, v0, w0, cp0, geometry)

    u_prime[:, 0] = u0.ravel()
    v_prime[:, 0] = v0.ravel()
    w_prime[:, 0] = w0.ravel()
    uvw_prime[:, 0] = np.concatenate((u0.ravel(), v0.ravel(), w0.ravel()))

    cp_prime[:, 0] = cp0.ravel()
    cp_mean_yz[:, :, 0] = cp_mean_yz_0

    # Remaining snapshots
    for ii, fname in enumerate(snapshots[1:], start=1):
        print(f"[{ii+1}/{m}] Processing {fname}")
        path = data_dir / fname
        u, v, w, cp, _, _ = load_fields(path)
        cp_mean_yz_i = subtract_means_inplace(u, v, w, cp, geometry)

        u_prime[:, ii] = u.ravel()
        v_prime[:, ii] = v.ravel()
        w_prime[:, ii] = w.ravel()
        uvw_prime[:, ii] = np.concatenate((u.ravel(), v.ravel(), w.ravel()))

        cp_prime[:, ii] = cp.ravel()
        cp_mean_yz[:, :, ii] = cp_mean_yz_i

    # Build output filename from iterations
    first_it = parse_iteration(snapshots[0])
    last_it = parse_iteration(snapshots[-1])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / (
        f"combined_snapshots_CASE_{case_num}_"
        f"{geometry}_ite_{first_it}-{last_it}.npz"
    )

    np.savez_compressed(
        save_file,
        u=u_prime,
        v=v_prime,
        w=w_prime,
        uvw=uvw_prime,
        cp_prime=cp_prime,
        cp_mean_yz=cp_mean_yz,
        y=y_1d,
        z=z_1d,
        nx=nx,
        ny=ny,
        nz=nz,
        geometry=geometry,
        case_num=case_num,
        first_it=first_it,
        last_it=last_it,
    )

    dt = time.time() - t0
    print("\nSaved snapshot matrices to:")
    print(f"  {save_file}")
    print(f"Total execution time: {dt:.2f} s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Build snapshot matrices (u', v', w', cp') from RHEA .h5 files."
    )
    p.add_argument(
        "--case-num",
        type=int,
        required=True,
        help="Case number (stored in the output file metadata).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the .h5 snapshot files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the .npz snapshot file will be written.",
    )
    p.add_argument(
        "--start-file",
        type=str,
        required=True,
        help="First .h5 file in the range (inclusive).",
    )
    p.add_argument(
        "--end-file",
        type=str,
        required=True,
        help="Last .h5 file in the range (inclusive).",
    )
    p.add_argument(
        "--geometry",
        type=str,
        choices=["channel", "duct"],
        default="channel",
        help="Flow geometry: 'channel' (homogeneous in x,z) or 'duct' (homogeneous in x).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_snapshots(
        case_num=args.case_num,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_file=args.start_file,
        end_file=args.end_file,
        geometry=args.geometry,
    )


if __name__ == "__main__":
    main()

