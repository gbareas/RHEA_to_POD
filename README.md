# RHEA_to_POD

Tools to convert **RHEA** DNS solver output (3D `*.h5` files) into
snapshot matrices suitable for **POD** (and later SPOD/DMD), and to
post-process the resulting modes.

## Current features

- `scripts/build_snapshots_matrix.py`:
  - Reads a series of RHEA `.h5` snapshots.
  - Removes ghost cells.
  - Computes and subtracts the appropriate mean:
    - `channel`: mean over x,z → function of y.
    - `duct`: mean over x → function of (y,z).
  - Builds flattened snapshot matrices for u', v', w', cp'.
  - Saves everything into a compressed `.npz` file.

Planned scripts:

- `scripts/compute_SVD.py` – compute SVD/POD from snapshot matrices.
- `scripts/post_process_SVD.py` – post-process modes, reconstructions, plots.


python scripts/build_snapshots_matrix.py \
  --case-num 80 \
  --data-dir /path/to/RHEA/h5/files \
  --output-dir /path/to/output \
  --start-file 3d_turbulent_channel_flow_7500000.h5 \
  --end-file   3d_turbulent_channel_flow_8400000.h5 \
  --geometry channel




