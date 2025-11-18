####################################################
## Post-process POD modes and plot colormaps #######
####################################################


#!/usr/bin/env python3
"""
Post-process POD modes: reconstruct 3D fields and plot Z-Y colormaps
for a given mode index (averaged in x).

Usage example
-------------
python scripts/post_process_SVD.py \
  --grid-h5 /path/to/reference_snapshot.h5 \
  --pod-u   /path/to/svd_u_CASE_28_ite_5221-6720.npz \
  --pod-cp  /path/to/svd_cp_CASE_28_ite_5221-6720.npz \
  --pod-T   /path/to/svd_T_CASE_28_ite_5221-6720.npz \
  --case-num 28 \
  --first-it 5221 \
  --last-it 6720 \
  --mode 1 \
  --output-dir /path/to/output/plots
"""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


plt.rc("text", usetex=True)
plt.rc("font", size=12)
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_grid(grid_file: Path):
    """
    Load grid coordinates x,y,z from a reference .h5 file and remove ghost cells.

    Returns
    -------
    x, y, z : arrays of shape (nx, ny, nz) without ghosts
    y_snap, z_snap : flattened Y,Z coordinates for ZY plane plots
    """
    with h5py.File(grid_file, "r") as f:
        x = f["x"][...]
        y = f["y"][...]
        z = f["z"][...]

    # Remove ghost cells
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))
    x = x[sl]
    y = y[sl]
    z = z[sl]

    # Build flattened Y,Z for ZY plane: we reuse your original logic
    # z_data: shape (nz, ny, nx) originally; after trimming same order.
    # For ZY plane we use x=0 slice.
    y_zy = y[0, :, :]      # shape (ny, nz) - originally (z,y,x)
    y_zy = np.swapaxes(y_zy, 0, 1)  # -> (nz, ny) -> (ny, nz) after interpretation
    y_snap = y_zy.flatten()

    z_zy = z[:, :, 0]      # shape (nz, ny)
    z_snap = z_zy.flatten()

    return x, y, z, y_snap, z_snap


def load_pod_modes(pod_file: Path, mode_index: int):
    """
    Load Phi from a POD .npz and return the 'mode_index'-th spatial mode.

    Parameters
    ----------
    pod_file : Path
        .npz file containing Phi, Sigma, Psi
    mode_index : int (1-based)
        Index of the mode to extract

    Returns
    -------
    phi_mode : 1D np.ndarray of length n_space
    """
    data = np.load(pod_file)
    if "Phi" in data:
        Phi = data["Phi"]
    elif "Phi_u" in data:
        Phi = data["Phi_u"]
    else:
        raise KeyError(
            f"No 'Phi' or 'Phi_u' key found in {pod_file}. Available: {list(data.keys())}"
        )

    if mode_index < 1 or mode_index > Phi.shape[1]:
        raise ValueError(
            f"Mode index {mode_index} out of range 1..{Phi.shape[1]} for file {pod_file}"
        )

    return Phi[:, mode_index - 1]


def plot_zy_colormap(
    z_snap,
    y_snap,
    field_2d,
    norm_factor,
    label,
    title_suffix,
    out_path: Path,
    cmap="RdGy",
):
    """
    Make a Z-Y colormap using tricontourf with consistent normalization.
    """
    field_flat = (field_2d / norm_factor).flatten()

    plt.clf()
    my_norm = colors.Normalize(vmin=np.min(field_flat), vmax=np.max(field_flat))

    # NOTE: using your scaling by 100e-6; adapt if needed
    cs = plt.tricontourf(
        z_snap / (100.0e-6),
        y_snap / (100.0e-6),
        field_flat,
        cmap=cmap,
        norm=my_norm,
    )

    cbar = plt.colorbar(cs, shrink=0.5, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    plt.text(4.5, 2.2, label, fontsize=9)

    plt.xlim(0.0, 4.2)
    plt.xticks(np.arange(0.0, 4.21, 0.5))
    plt.tick_params(
        axis="x",
        left=True,
        right=True,
        top=True,
        bottom=True,
        direction="inout",
        labelsize=9,
    )

    plt.ylim(0.0, 2.0)
    plt.yticks(np.arange(0.0, 2.01, 1.0))
    plt.tick_params(
        axis="y",
        left=True,
        right=True,
        top=True,
        bottom=True,
        direction="inout",
        labelsize=9,
    )

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", pad=2.5)
    plt.xlabel(r"$z / \delta$", size=9)
    plt.ylabel(r"$y / \delta$", size=9)
    plt.title(title_suffix, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved colormap to {out_path}")


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-process POD modes and plot Z-Y colormaps (averaged in x)."
    )
    p.add_argument("--grid-h5", type=Path, required=True, help="Reference .h5 file.")
    p.add_argument("--pod-u", type=Path, required=True, help="POD file for u.")
    p.add_argument("--pod-cp", type=Path, required=True, help="POD file for cp.")
    p.add_argument("--pod-T", type=Path, required=True, help="POD file for T.")
    p.add_argument("--case-num", type=int, required=True, help="Case number.")
    p.add_argument("--first-it", type=int, required=True, help="First iteration.")
    p.add_argument("--last-it", type=int, required=True, help="Last iteration.")
    p.add_argument(
        "--mode",
        type=int,
        required=True,
        help="POD mode index (1-based) to plot.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory for output plots.",
    )
    p.add_argument(
        "--u-bulk",
        type=float,
        default=1.0,
        help="Normalization velocity u_b (default: 1.0).",
    )
    p.add_argument(
        "--cp-ref",
        type=float,
        default=5216.1,
        help="Reference cp value for normalization (default: 5216.1).",
    )
    p.add_argument(
        "--T-ref",
        type=float,
        default=300.0,
        help="Reference T value for normalization (default: 300.0).",
    )
    p.add_argument(
        "--save-h5",
        action="store_true",
        help="If set, also save a .h5 file with 3D modes and grid.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Grid
    x, y, z, y_snap, z_snap = load_grid(args.grid_h5)
    nx, ny, nz = x.shape
    n_space = nx * ny * nz
    print(f"Grid (no ghosts): nx={nx}, ny={ny}, nz={nz}, n_space={n_space}")

    # Load modes
    phi_u = load_pod_modes(args.pod_u, args.mode)
    phi_cp = load_pod_modes(args.pod_cp, args.mode)
    phi_T = load_pod_modes(args.pod_T, args.mode)

    # Reshape to 3D fields
    u_mode = phi_u.reshape((nx, ny, nz))
    cp_mode = phi_cp.reshape((nx, ny, nz))
    T_mode = phi_T.reshape((nx, ny, nz))

    # Average over x to get Z-Y planes
    u_zy = np.mean(u_mode, axis=0)   # (ny, nz)
    cp_zy = np.mean(cp_mode, axis=0)
    T_zy = np.mean(T_mode, axis=0)

    # For ZY plane, swap axes like original script did
    u_zy = np.swapaxes(u_zy, 0, 1)   # (nz, ny)
    cp_zy = np.swapaxes(cp_zy, 0, 1)
    T_zy = np.swapaxes(T_zy, 0, 1)

    # Output paths
    base_name = f"Case{args.case_num}_ite_{args.first_it}-{args.last_it}_mode{args.mode}"
    out_u = args.output_dir / "u_colormap" / f"{base_name}_u.png"
    out_cp = args.output_dir / "cp_colormap" / f"{base_name}_cp.png"
    out_T = args.output_dir / "T_colormap" / f"{base_name}_T.png"

    # Plots
    plot_zy_colormap(
        z_snap,
        y_snap,
        u_zy,
        norm_factor=args.u_bulk,
        label=r"$u^{\prime} / u_b$",
        title_suffix=rf"$u'$ mode {args.mode}",
        out_path=out_u,
    )

    plot_zy_colormap(
        z_snap,
        y_snap,
        cp_zy,
        norm_factor=args.cp_ref,
        label=r"${cp}^{\prime} / {cp}_{ref}$",
        title_suffix=rf"$c_p'$ mode {args.mode}",
        out_path=out_cp,
    )

    plot_zy_colormap(
        z_snap,
        y_snap,
        T_zy,
        norm_factor=args.T_ref,
        label=r"$T^{\prime} / T_{ref}$",
        title_suffix=rf"$T'$ mode {args.mode}",
        out_path=out_T,
    )

    # Optional HDF5 export of 3D modes + grid
    if args.save_h5:
        h5_out = args.output_dir / f"{base_name}_POD_modes.h5"
        h5_out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_out, "w") as f:
            f.create_dataset("X", data=x)
            f.create_dataset("Y", data=y)
            f.create_dataset("Z", data=z)
            f.create_dataset("u_mode", data=u_mode)
            f.create_dataset("cp_mode", data=cp_mode)
            f.create_dataset("T_mode", data=T_mode)
        print(f"Saved HDF5 with 3D modes to {h5_out}")

    dt = time.time() - t0
    print(f"Post-processing completed in {dt:.2f} s")


if __name__ == "__main__":
    main()

