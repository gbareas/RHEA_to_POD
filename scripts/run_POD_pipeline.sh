#!/bin/bash

# 1. Safety First: Stop immediately if any command fails
set -e

# 2. Get the directory of this script (so it works from anywhere)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- CONFIGURATION VARIABLES ---------------------------------------------
CASE_NUM="3"
GEOMETRY="channel"
VARIABLE="u"  # For the SVD step

# Iterations (used for filenames)
START_IT="29000000"
END_IT="31175000"

# Filenames (constructed from iterations)
START_FILE="3d_HP_MFU_Case${CASE_NUM}_${START_IT}.h5"
END_FILE="3d_HP_MFU_Case${CASE_NUM}_${END_IT}.h5"

# Directories
DATA_DIR="path/to_file" # Add path to the raw data extracted from RHEA
OUTPUT_DIR="path/to_snapshots_matrix"

# Dynamic Filenames (Input for SVD = Output of Build)
SNAPSHOT_MATRIX="${OUTPUT_DIR}/snapshot_matrices/Case${CASE_NUM}/combined_snapshots_CASE_${CASE_NUM}_${GEOMETRY}_ite_${START_IT}-${END_IT}.npz"
SVD_OUTPUT="${OUTPUT_DIR}/SVD_output/Case${CASE_NUM}/svd_${VARIABLE}_CASE_${CASE_NUM}_${GEOMETRY}_ite_${START_IT}-${END_IT}.npz"
# -------------------------------------------------------------------------

echo "========================================================="
echo " PROCESSING CASE ${CASE_NUM} (${START_IT} - ${END_IT})"
echo "========================================================="

# CHECK: Does the matrix file already exist?
if [ -f "$SNAPSHOT_MATRIX" ]; then
    echo ""
    echo ">>> SKIP STEP 1: Snapshot matrix found."
    echo "    Using existing file: $SNAPSHOT_MATRIX"
else
    echo ""
    echo ">>> STEP 1: Building Snapshot Matrix..."
    python3 "$SCRIPT_DIR/build_snapshots_matrix.py" \
        --case-num "$CASE_NUM" \
        --geometry "$GEOMETRY" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --start-file "$START_FILE" \
        --end-file "$END_FILE"
fi

echo ""
echo ">>> STEP 2: Computing SVD for variable '${VARIABLE}'..."
python3 "$SCRIPT_DIR/compute_SVD.py" \
    --input "$SNAPSHOT_MATRIX" \
    --output "$SVD_OUTPUT" \
    --variable "$VARIABLE"

echo ""
echo "✅ Pipeline Complete!"
