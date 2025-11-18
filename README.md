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


## Instructions to run RHEA to POD


### Building snapshot matrices from RHEA DNS output

To generate the snapshot matrices from a sequence of RHEA `.h5` files:

```bash
python scripts/build_snapshots_matrix.py \
  --case-num 80 \
  --data-dir /path/to/h5_files \
  --output-dir /path/to/output \
  --start-file 3d_turbulent_channel_flow_7500000.h5 \
  --end-file   3d_turbulent_channel_flow_8400000.h5 \
  --geometry channel


### Compute POD (SVD) from snapshot matrices 

After building the snapshot matrices, you can compute the POD modes, eigenvalues and temporal coefficients via SVD using:
- `scripts/compute_SVD.py`


- Loads a snapshot matrix from a `.npz` file (e.g. `u`, `v`, `w`, `uvw`, `cp_prime`).
- Computes the (economy) SVD:
  \[
  \mathbf{X} = \mathbf{\Phi}\,\mathbf{\Sigma}\,\mathbf{\Psi}^{T}
  \]
  where \(\mathbf{X}\) is the snapshot matrix.
- Stores the spatial modes \(\Phi\), singular values \(\Sigma\), and temporal
  coefficients \(\Psi\) in a compressed `.npz` file.


```bash
python scripts/compute_SVD.py \
  --input  /path/to/combined_snapshots_CASE_28_channel_ite_64385000-68112500.npz \
  --output /path/to/svd_u_CASE_28_channel_ite_64385000-68112500.npz \
  --variable u


## Post-process POD modes and generate colormaps

Once the snapshot matrices and POD (SVD) have been computed, you can reconstruct and visualize the dominant spatial structures from the POD modes using:


- `scripts/post_process_SVD.py`


- Loads the grid from a reference RHEA `.h5` file.
- Loads POD results (`Phi`) for velocity, heat capacity and temperature.
- Extracts a given mode index, reshapes it into a 3D field, and averages over the
  streamwise direction to obtain a Z–Y plane.
- Produces Z–Y colormaps of the selected mode for:
  - streamwise velocity fluctuations \(u'\),
  - heat capacity fluctuations \(c_p'\),
  - temperature fluctuations \(T'\).
- Optionally saves the full 3D mode fields and grid into a new `.h5` file.


```bash
python scripts/post_process_SVD.py \
  --grid-h5 /path/to/reference_snapshot.h5 \
  --pod-u   /path/to/svd_u_CASE_28_ite_64385000-68112500.npz \
  --pod-cp  /path/to/svd_cp_CASE_28_ite_64385000-68112500.npz \
  --pod-T   /path/to/svd_T_CASE_28_ite_64385000-68112500.npz \
  --case-num 28 \
  --first-it 64385000 \
  --last-it  68112500 \
  --mode 1 \
  --output-dir /path/to/output/plots \
  --u-bulk 1.0 \
  --cp-ref 5216.1 \
  --T-ref 300.0 \
  --save-h5

