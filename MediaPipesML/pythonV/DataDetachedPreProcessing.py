"""
Processes raw landmarks from raw files to test new derivations of features while preserving orginial data.

Input:  data/custom_gesture_data_rock.csv
        data/custom_gesture_data_paper.csv
        data/custom_gesture_data_scissors.csv

Output: data/processed/custom_gesture_data_rock.csv
        data/processed/custom_gesture_data_paper.csv
        data/processed/custom_gesture_data_scissors.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR   = "./data"
OUTPUT_DIR = "./data/processed"

FILES = [
    "custom_gesture_data_rock.csv",
    "custom_gesture_data_paper.csv",
    "custom_gesture_data_scissors.csv",
]

REFERENCE_PALM_SIZE = 0.14

# =============================================================================
# FEATURE COMPUTATION — mirrors features.cpp exactly
# =============================================================================

def get_landmark_distance(l1, l2, x, y):
    dx = x[l1] - x[l2]
    dy = y[l1] - y[l2]
    return np.sqrt(dx*dx + dy*dy)


def get_finger_vector(l1, l2, x, y):
    return np.array([x[l2] - x[l1], y[l2] - y[l1]], dtype=np.float64)


def get_angle_between(v1, v2):
    dot   = v1[0]*v2[0] + v1[1]*v2[1]
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    norms = norm1 * norm2
    if norms == 0.0:
        return 0.0
    return np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))

def get_dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def normalize_vec(v):

    mag = np.linalg.norm(v)
    if mag > 0.0:
        return v / mag
    return v


def compute_features_from_landmarks(x, y):
    """
    Given 21 landmark x and y coords (normalized 0-1),
    recompute all features exactly as C++ features.cpp does.
    Returns a dict, or None if palm_size is 0.
    """
    palm_size = get_landmark_distance(9, 0, x, y)
    if palm_size == 0.0:
        return None

    scale = REFERENCE_PALM_SIZE / palm_size

    # Raw direction vectors (same landmark pairs as C++)
    V_index       = get_finger_vector(7,  6,  x, y)
    V_middle      = get_finger_vector(11, 10, x, y)
    V_ring        = get_finger_vector(15, 14, x, y)
    V_pinky       = get_finger_vector(19, 18, x, y)
    V_baseline1   = get_finger_vector(1,  0,  x, y)
    V_middle_base = get_finger_vector(10, 9,  x, y)
    V_ring_base   = get_finger_vector(14, 13, x, y)

    # Normalize to unit length — matches C++ normalize block
    dirV_index_n       = normalize_vec(V_index)
    dirV_middle_n      = normalize_vec(V_middle)
    dirV_ring_n        = normalize_vec(V_ring)
    dirV_pinky_n       = normalize_vec(V_pinky)
    dirV_baseline1_n   = normalize_vec(V_baseline1)
    dirV_middle_base_n = normalize_vec(V_middle_base)
    dirV_ring_base_n   = normalize_vec(V_ring_base)

    # Angles — computed on normalized vectors, matches C++
    angle_index               = get_angle_between(dirV_index_n,      dirV_baseline1_n)
    angle_middle              = get_angle_between(dirV_middle_n,      dirV_baseline1_n)
    angle_ring                = get_angle_between(dirV_ring_n,        dirV_baseline1_n)
    angle_pinky               = get_angle_between(dirV_pinky_n,       dirV_baseline1_n)
    angle_baseline1           = get_angle_between(dirV_baseline1_n,   dirV_baseline1_n)  # always 0
    angle_middle_to_ring      = get_angle_between(dirV_middle_n,      dirV_ring_n)
    angle_base_middle_to_ring = get_angle_between(dirV_middle_base_n, dirV_ring_base_n)
    angle_middle_to_baseline1 = get_angle_between(dirV_middle_n,      dirV_baseline1_n)

    # Distances — scaled by palm size
    dis_index  = get_landmark_distance(8,  0, x, y) * scale
    dis_middle = get_landmark_distance(12, 0, x, y) * scale
    dis_ring   = get_landmark_distance(16, 0, x, y) * scale
    dis_pinky  = get_landmark_distance(20, 0, x, y) * scale

    # Aggregated pairwise distance — matches C++ getDistancesAggregated()
    coords = np.column_stack([x, y])
    dist_aggregated = pdist(coords, metric='euclidean').mean() * scale

    return {
        'angle_index':               angle_index,
        'angle_middle':              angle_middle,
        'angle_ring':                angle_ring,
        'angle_pinky':               angle_pinky,
        'angle_baseline1':           angle_baseline1,
        'angle_middle_to_ring':      angle_middle_to_ring,
        'angle_base_middle_to_ring': angle_base_middle_to_ring,
        'angle_middle_to_baseline1': angle_middle_to_baseline1,
        'dis_index_tip_to_base':     dis_index,
        'dis_middle_tip_to_base':    dis_middle,
        'dis_ring_tip_to_base':      dis_ring,
        'dis_pinky_tip_to_base':     dis_pinky,
        'distancesAggregated':       dist_aggregated,
        'dirV_index_x':              dirV_index_n[0],
        'dirV_index_y':              dirV_index_n[1],
        'dirV_middle_x':             dirV_middle_n[0],
        'dirV_middle_y':             dirV_middle_n[1],
        'dirV_ring_x':               dirV_ring_n[0],
        'dirV_ring_y':               dirV_ring_n[1],
        'dirV_pinky_x':              dirV_pinky_n[0],
        'dirV_pinky_y':              dirV_pinky_n[1],
        'dirV_middle_base_x':        dirV_middle_base_n[0],
        'dirV_middle_base_y':        dirV_middle_base_n[1],
        'dirV_ring_base_x':          dirV_ring_base_n[0],
        'dirV_ring_base_y':          dirV_ring_base_n[1],
        'dot_index_middle':          get_dot_product(V_index, V_middle),
        'dot_index_ring':            get_dot_product(V_index, V_ring),
        'dot_index_pinky':           get_dot_product(V_index, V_pinky),
        'dot_middle_ring':           get_dot_product(V_middle, V_ring),
        'dot_middle_pinky':          get_dot_product(V_middle, V_pinky),
        'dot_ring_pinky':            get_dot_product(V_ring, V_pinky),
    }


# =============================================================================
# PROCESS CSV
# =============================================================================

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"  Rows loaded: {len(df):,}")

    # Verify landmark columns exist
    x_cols = [f'landmark_{i}_x' for i in range(21)]
    y_cols = [f'landmark_{i}_y' for i in range(21)]
    missing = [c for c in x_cols + y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing landmark columns: {missing[:4]}...")

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        x = row[x_cols].values.astype(np.float64)
        y = row[y_cols].values.astype(np.float64)

        features = compute_features_from_landmarks(x, y)
        if features is None:
            skipped += 1
            continue

        out_row = {
            'timestamp':    row['timestamp'],
            'gesture_type': row['gesture_type'],
        }
        out_row.update(features)
        rows.append(out_row)

    print(f"  Rows processed: {len(rows):,}  |  Skipped (zero palm): {skipped}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Sanity check
    print("  Verification — mean vector magnitudes (should all be ~1.0):")
    for vec in ['dirV_index', 'dirV_middle', 'dirV_ring',
                'dirV_pinky', 'dirV_middle_base', 'dirV_ring_base']:
        x_vals = out_df[f'{vec}_x'].values
        y_vals = out_df[f'{vec}_y'].values
        mags = np.sqrt(x_vals**2 + y_vals**2)
        print(f"    {vec:20s}  mean={mags.mean():.6f}  min={mags.min():.6f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in FILES:
        input_path  = os.path.join(DATA_DIR,   filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(input_path):
            print(f"SKIPPING (not found): {input_path}")
            continue

        print(f"\nProcessing: {filename}")
        process_file(input_path, output_path)

    print("\nDone. Set DATA_DIR = './data/processed' in gesture_classification_train.py and retrain.")


if __name__ == "__main__":
    main()