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

"""
Post-process POD modes: reconstruct 3D fields and plot Z-Y colormaps.
"""

#!/usr/bin/env python3
"""
Post-process POD modes: reconstruct 3D fields and plot Z-Y colormaps.
Now supports optional inputs for cp and T.
"""

import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

# Latex configuration (Safe Fallback)
try:
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=12)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
except Exception:
    print("Warning: LaTeX not found or configuration failed. Using standard fonts.")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_grid(grid_file: Path):
    """
    Load grid coordinates x,y,z, transpose to (x,y,z), remove ghosts.
    Returns 3D arrays and 2D arrays for the Z-Y plane.
    """
    with h5py.File(grid_file, "r") as f:
        # CRITICAL FIX: Transpose to (x, y, z) to match build_snapshots.py
        # Original data is likely (z, y, x)
        x = np.transpose(f["x"][...], (2, 1, 0))
        y = np.transpose(f["y"][...], (2, 1, 0))
        z = np.transpose(f["z"][...], (2, 1, 0))

    # Remove ghost cells
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))
    x, y, z = x[sl], y[sl], z[sl]

    # Extract 2D plane coordinates for Z-Y plots (x-averaged view)
    # y is (nx, ny, nz) -> take slice at x=0 -> (ny, nz)
    # We want Z on horizontal (x-axis of plot) and Y on vertical (y-axis of plot)
    y_zy = y[0, :, :] 
    z_zy = z[0, :, :]

    return x, y, z, y_zy, z_zy


def load_pod_mode_column(pod_file: Path, mode_index: int):
    """
    Load a specific mode column efficiently using memory mapping.
    """
    # FIX: mmap_mode='r' prevents loading the whole file into RAM
    with np.load(pod_file, mmap_mode='r') as data:
        keys = list(data.keys())
        
        # Smart key detection (Phi, Phi_u, Phi_cp, etc.)
        phi_key = next((k for k in keys if "Phi" in k), None)
        if not phi_key:
             raise KeyError(f"No 'Phi' key found in {pod_file}. Available: {keys}")

        Phi = data[phi_key]
        
        # Check bounds
        if mode_index < 1 or mode_index > Phi.shape[1]:
            raise ValueError(f"Mode {mode_index} out of bounds (Max: {Phi.shape[1]}).")
        
        # Copy only the specific column to memory
        return np.array(Phi[:, mode_index - 1])


def plot_zy_colormap(
    z_grid,     # 2D array (ny, nz)
    y_grid,     # 2D array (ny, nz)
    field_2d,   # 2D array (ny, nz)
    norm_factor,
    label,
    title_suffix,
    out_path: Path,
    cmap="RdGy",
):
    """
    Make a Z-Y colormap using contourf (structured).
    """
    field_norm = field_2d / norm_factor
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Normalize centered on 0 for fluctuation fields
    val_max = np.max(np.abs(field_norm))
    # Avoid zero division if field is empty
    if val_max == 0: val_max = 1e-10
    
    my_norm = colors.Normalize(vmin=-val_max, vmax=val_max)

    # Use contourf for structured data
    # z_grid goes to X-axis, y_grid goes to Y-axis
    cs = ax.contourf(
        z_grid / 100.0e-6,
        y_grid / 100.0e-6,
        field_norm,
        levels=128,
        cmap=cmap,
        norm=my_norm,
    )

    cbar = plt.colorbar(cs, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    cbar.locator = ticker.MaxNLocator(nbins=5)
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar.update_ticks()

    # Dynamic limits based on grid content
    z_max = np.max(z_grid) / 100.0e-6
    y_max = np.max(y_grid) / 100.0e-6
    
    ax.set_xlim(0.0, z_max)
    ax.set_ylim(0.0, y_max)
    
    ax.set_xlabel(r"$z / \delta$", size=10)
    ax.set_ylabel(r"$y / \delta$", size=10)
    ax.set_title(title_suffix, fontsize=11)
    ax.set_aspect("equal", adjustable="box")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig) # Close to free memory
    print(f"Saved colormap to {out_path}")


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-process POD modes and plot Z-Y colormaps."
    )
    # Required
    p.add_argument("--grid-h5", type=Path, required=True, help="Reference .h5 file.")
    p.add_argument("--pod-u", type=Path, required=True, help="POD file for u.")
    p.add_argument("--case-num", type=int, required=True)
    p.add_argument("--first-it", type=int, required=True)
    p.add_argument("--last-it", type=int, required=True)
    p.add_argument("--mode", type=int, required=True, help="Mode index (1-based).")
    p.add_argument("--output-dir", type=Path, required=True)

    # Optional inputs (default to None)
    p.add_argument("--pod-cp", type=Path, default=None, required=False, help="POD file for cp.")
    p.add_argument("--pod-T", type=Path, default=None, required=False, help="POD file for T.")

    # Constants
    p.add_argument("--u-bulk", type=float, default=1.0)
    p.add_argument("--cp-ref", type=float, default=5216.1)
    p.add_argument("--T-ref", type=float, default=300.0)
    p.add_argument("--save-h5", action="store_true")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # 1. Load Grid
    print(f"Loading grid: {args.grid_h5}")
    x, y, z, y_zy, z_zy = load_grid(args.grid_h5)
    nx, ny, nz = x.shape
    print(f"Grid (corrected): nx={nx}, ny={ny}, nz={nz}")

    # Prepare naming
    base_name = f"Case{args.case_num}_ite_{args.first_it}-{args.last_it}_mode{args.mode}"

    # --- PROCESS U (Mandatory) ---
    print(f"Processing U mode {args.mode}...")
    phi_u = load_pod_mode_column(args.pod_u, args.mode)
    u_mode = phi_u.reshape((nx, ny, nz))
    u_mean = np.mean(u_mode, axis=0) # Average over x -> (ny, nz)

    plot_zy_colormap(
        z_zy, y_zy, u_mean, 
        args.u_bulk, r"$u^{\prime} / U_b$", rf"$u'$ $mode$ {args.mode}",
        args.output_dir / "u_colormap" / f"{base_name}_u.png"
    )

    # --- PROCESS CP (Optional) ---
    cp_mode = None
    if args.pod_cp is not None:
        print(f"Processing CP mode {args.mode}...")
        phi_cp = load_pod_mode_column(args.pod_cp, args.mode)
        cp_mode = phi_cp.reshape((nx, ny, nz))
        cp_mean = np.mean(cp_mode, axis=0)

        plot_zy_colormap(
            z_zy, y_zy, cp_mean, 
            args.cp_ref, r"${cp}^{\prime} / {cp}_{ref}$", rf"$c_p'$ mode {args.mode}",
            args.output_dir / "cp_colormap" / f"{base_name}_cp.png"
        )

    # --- PROCESS T (Optional) ---
    T_mode = None
    if args.pod_T is not None:
        print(f"Processing T mode {args.mode}...")
        phi_T = load_pod_mode_column(args.pod_T, args.mode)
        T_mode = phi_T.reshape((nx, ny, nz))
        T_mean = np.mean(T_mode, axis=0)

        plot_zy_colormap(
            z_zy, y_zy, T_mean, 
            args.T_ref, r"$T^{\prime} / T_{ref}$", rf"$T'$ mode {args.mode}",
            args.output_dir / "T_colormap" / f"{base_name}_T.png"
        )

    # --- SAVE H5 (Optional) ---
    if args.save_h5:
        h5_out = args.output_dir / f"{base_name}_POD_modes.h5"
        print(f"Exporting 3D modes to {h5_out}...")
        with h5py.File(h5_out, "w") as f:
            f.create_dataset("X", data=x)
            f.create_dataset("Y", data=y)
            f.create_dataset("Z", data=z)
            
            f.create_dataset("u_mode", data=u_mode)
            
            if cp_mode is not None:
                f.create_dataset("cp_mode", data=cp_mode)
            
            if T_mode is not None:
                f.create_dataset("T_mode", data=T_mode)
        
        print("Export complete.")

    dt = time.time() - t0
    print(f"Total time: {dt:.2f} s")


if __name__ == "__main__":
    main()