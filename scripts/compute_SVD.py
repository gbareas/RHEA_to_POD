####################################################
## Compute SVD/POD from snapshot matrices ##########
####################################################

"""
Compute SVD-based POD from snapshot matrices stored in a .npz file.

Typical usage
-------------
From the root of the RHEA_to_POD repo:

    python scripts/compute_SVD.py \
        --input  /path/to/combined_snapshots_CASE_28_channel_ite_64385000-68112500.npz \
        --output /path/to/svd_u_CASE_28_channel_ite_64385000-68112500.npz \
        --variable u

This script expects the input .npz to contain a 2D snapshot matrix, e.g.:

    - u       : shape (n_space, n_snapshots)
    - v, w    : same
    - uvw     : shape (3 * n_space, n_snapshots)
    - cp_prime: shape (n_space, n_snapshots)

It then computes the (economy) SVD:

    X = Phi @ diag(Sigma) @ Psi^T

and stores Phi, Sigma, Psi in a compressed .npz file.
"""

import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute SVD/POD from snapshot matrices stored in a .npz file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .npz file with snapshot matrices (e.g. u, v, w, uvw, cp_prime).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file to store POD (Phi, Sigma, Psi).",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="u",
        help=(
            "Key in the input .npz to use as snapshot matrix. "
            "Common options: 'u', 'v', 'w', 'uvw', 'cp_prime'. "
            "Default: 'u'."
        ),
    )
    parser.add_argument(
        "--full-matrices",
        action="store_true",
        help="Use full SVD (default: economy SVD with full_matrices=False).",
    )
    return parser.parse_args()


def compute_pod(
    input_file: Path,
    variable: str = "u",
    full_matrices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load snapshot matrix from input_file[variable] and compute SVD/POD.

    Parameters
    ----------
    input_file : Path
        Path to the .npz file containing the snapshot matrix.
    variable : str
        Key inside the .npz file to use as the snapshot matrix.
        Must correspond to a 2D array of shape (n_space, n_snapshots).
    full_matrices : bool
        If True, compute full SVD; otherwise, compute economy SVD.

    Returns
    -------
    Phi : np.ndarray
        Left singular vectors (POD modes in space), shape (n_space, r).
    Sigma : np.ndarray
        Singular values, shape (r,).
    PsiT : np.ndarray
        Right singular vectors transposed, shape (r, n_snapshots).
    """
    data = np.load(input_file)

    if variable not in data:
        raise KeyError(
            f"Variable {variable!r} not found in {input_file.name}. "
            f"Available keys: {list(data.keys())}"
        )

    X = data[variable]
    if X.ndim != 2:
        raise ValueError(
            f"Snapshot matrix '{variable}' must be 2D (n_space, n_snapshots); "
            f"got shape {X.shape}"
        )

    print(f"Loaded snapshot matrix '{variable}' from {input_file}")
    print(f"Matrix shape: n_space={X.shape[0]}, n_snapshots={X.shape[1]}")

    print("Starting SVD...")
    Phi, Sigma, PsiT = np.linalg.svd(X, full_matrices=full_matrices)
    print("SVD completed.")

    return Phi, Sigma, PsiT


def save_pod(
    output_file: Path,
    Phi: np.ndarray,
    Sigma: np.ndarray,
    PsiT: np.ndarray,
) -> None:
    """
    Save SVD/POD factors to a compressed NPZ file.

    Parameters
    ----------
    output_file : Path
        Path to the .npz file to create.
    Phi : np.ndarray
        Left singular vectors.
    Sigma : np.ndarray
        Singular values.
    PsiT : np.ndarray
        Right singular vectors transposed.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, Phi=Phi, Sigma=Sigma, Psi=PsiT)
    print(f"POD data saved to {output_file}")


def main() -> None:
    args = parse_args()
    start_time = time.time()

    Phi, Sigma, PsiT = compute_pod(
        input_file=args.input,
        variable=args.variable,
        full_matrices=args.full_matrices,
    )
    save_pod(args.output, Phi, Sigma, PsiT)

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

