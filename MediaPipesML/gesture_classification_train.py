"""
Rock Paper Scissors - LightGBM Classifier with Sliding Window
=============================================================
- Input:  3 CSV files, one per gesture (rock, paper, scissors)
          e.g. custom_gesture_data_rock.csv, _paper.csv, _scissors.csv
- Output: Trained LightGBM model saved to disk + evaluation plots
- Window: 15 frames (~0.5s at 30fps)
- Classes: 0 = rock, 1 = paper, 2 = scissors

Usage:
    1. Set DATA_DIR and FILE_MAP below to match your file locations
    2. Run: python train_classifier.py

Install dependencies:
    pip install lightgbm scikit-learn pandas numpy matplotlib joblib
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# =============================================================================
# CONFIGURATION — Edit these to match your setup
# =============================================================================

DATA_DIR  = "./data"   # Folder containing your CSV files

# Map each class label to its CSV filename (just the filename, not full path)
FILE_MAP = {
    0: "custom_gesture_data_rock.csv",
    1: "custom_gesture_data_paper.csv",
    2: "custom_gesture_data_scissors.csv",
}

LABEL_COLUMN  = "gesture_type"   # Column name for the label in your CSVs
WINDOW_SIZE   = 5             # Sliding window in frames (~0.5s at 30fps)
TEST_SIZE     = 0.2               # 20% of each file held out for testing
MODEL_OUT     = "rps_lgbm_model.pkl"

# Columns to DROP before training (not useful as features)
DROP_COLUMNS = ["timestamp", "gesture_type"]

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        40,
    "max_depth":         -1,
    "min_child_samples": 10,
    "subsample":         0.8,  
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "verbose":           -1,
    "random_state":      42,
}

# =============================================================================
# STEP 1: Load CSVs — one per gesture class
# =============================================================================

def load_gesture_csvs(data_dir: str, file_map: dict) -> dict[int, pd.DataFrame]:
    """
    Load one CSV per gesture class.
    Returns a dict: { class_label: DataFrame }
    """
    dfs = {}
    print("Loading gesture CSV files:")
    for label, filename in file_map.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for class {label}: {path}")
        df = pd.read_csv(path)
        
        
        df = df.drop(columns=[c for c in df.columns if c.startswith("landmark")], errors="ignore")

        print(f"  Class {label}: {filename:45s}  {len(df):>7,} rows")
        dfs[label] = df
    return dfs


# =============================================================================
# STEP 2: Build sliding windows — per class file, then split train/test
# =============================================================================

def make_windows(feature_array: np.ndarray, label: int, window_size: int):
    """
    Flatten a rolling window of `window_size` rows into one feature vector.
    Label is applied to every window (all rows in this file share the same class).

    Returns:
        X: shape (n_samples, n_features * window_size)
        y: shape (n_samples,)
    """
    n_rows, n_features = feature_array.shape
    n_samples = n_rows - window_size + 1

    X = np.empty((n_samples, n_features * window_size), dtype=np.float64)
    for i in range(n_samples):
        X[i] = feature_array[i : i + window_size].flatten()

    y = np.full(n_samples, label, dtype=np.int32)
    return X, y


def build_train_test_sets(dfs: dict, drop_cols: list, window_size: int, test_size: float):
    """
    For each gesture CSV:
      1. Drop non-feature columns
      2. Split rows 80/20 by time order (no shuffle — preserves temporal integrity)
      3. Build sliding windows on train portion and test portion separately

    Then concatenate all classes together and shuffle the train set.
    """
    X_train_parts, y_train_parts = [], []
    X_test_parts,  y_test_parts  = [], []

    print(f"\nBuilding {window_size}-frame sliding windows (80/20 time-ordered split per class):")

    for label, df in dfs.items():
        # Drop non-feature columns
        feature_cols = [c for c in df.columns if c not in drop_cols]
        features = df[feature_cols].values.astype(np.float64)

        # Time-ordered split — no shuffle
        split_at = int(len(features) * (1 - test_size))
        train_features = features[:split_at]
        test_features  = features[split_at:]

        # Build windows
        X_tr, y_tr = make_windows(train_features, label, window_size)
        X_te, y_te = make_windows(test_features,  label, window_size)

        X_train_parts.append(X_tr)
        y_train_parts.append(y_tr)
        X_test_parts.append(X_te)
        y_test_parts.append(y_te)

        print(f"  Class {label}: {len(train_features):>6,} train rows -> {len(X_tr):>6,} windows  |  "
              f"{len(test_features):>6,} test rows -> {len(X_te):>6,} windows")

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_test  = np.concatenate(X_test_parts,  axis=0)
    y_test  = np.concatenate(y_test_parts,  axis=0)

    # Shuffle train set so model doesn't see all rock, then all paper, then all scissors
    rng = np.random.default_rng(42)
    train_shuffle = rng.permutation(len(X_train))
    X_train = X_train[train_shuffle]
    y_train = y_train[train_shuffle]

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 3: Train LightGBM
# =============================================================================

def train_model(X_train, y_train, X_test, y_test, params: dict) -> lgb.LGBMClassifier:
    print(f"\nTraining LightGBM...")
    print(f"  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples  |  Features: {X_train.shape[1]}")

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    return model


# =============================================================================
# STEP 4: Evaluate
# =============================================================================

def evaluate(model, X_test, y_test, base_feature_names: list):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rock (0)", "Paper (1)", "Scissors (2)"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rock", "Paper", "Scissors"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix  -  Accuracy: {acc*100:.2f}%")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to: confusion_matrix.png")

    # Feature importance (top 30) with human-readable labels
    # t-0 = most recent frame in the window, t-(WINDOW_SIZE-1) = oldest
    windowed_feature_names = [
        f"{name} (t-{WINDOW_SIZE - 1 - frame_offset})"
        for frame_offset in range(WINDOW_SIZE)
        for name in base_feature_names
    ]

    importance = model.feature_importances_
    top_n = min(30, len(importance))
    top_idx = np.argsort(importance)[-top_n:][::-1]
    top_labels = [windowed_feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(range(top_n), importance[top_idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Feature importance saved to:  feature_importance.png")

    return acc


# =============================================================================
# STEP 5: Save model
# =============================================================================

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")


# =============================================================================
# INFERENCE HELPER — use this after training to predict on new recordings
# =============================================================================

GESTURE_NAMES = {0: "Rock", 1: "Paper", 2: "Scissors"}

def predict_on_new_csv(model_path: str, csv_path: str, drop_cols: list, window_size: int) -> pd.DataFrame:
    """
    Load a saved model and run predictions on a new CSV file.
    Returns a DataFrame with a prediction column per frame.
    Frames before window_size-1 will have prediction = -1 (not enough history yet).

    Example:
        results = predict_on_new_csv("rps_lgbm_model.pkl", "new_session.csv", DROP_COLUMNS, WINDOW_SIZE)
        print(results.head())
    """
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    features = df[feature_cols].values.astype(np.float64)

    n_rows, n_features = features.shape
    n_samples = n_rows - window_size + 1

    X = np.empty((n_samples, n_features * window_size), dtype=np.float64)
    for i in range(n_samples):
        X[i] = features[i : i + window_size].flatten()

    preds = model.predict(X).astype(int)
    pred_labels = [GESTURE_NAMES[p] for p in preds]

    result = pd.DataFrame({
        "frame":      range(n_rows),
        "prediction": [-1]        * (window_size - 1) + list(preds),
        "gesture":    ["unknown"] * (window_size - 1) + pred_labels,
    })
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # 1. Load
    dfs = load_gesture_csvs(DATA_DIR, FILE_MAP)

    # 2. Build windowed train/test sets
    X_train, X_test, y_train, y_test = build_train_test_sets(
        dfs, DROP_COLUMNS, WINDOW_SIZE, TEST_SIZE
    )

    print(f"\nFinal dataset:")
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"  Train label dist: { {v: int((y_train==v).sum()) for v in [0,1,2]} }")
    print(f"  Test  label dist: { {v: int((y_test==v).sum())  for v in [0,1,2]} }")

    # 3. Train
    model = train_model(X_train, y_train, X_test, y_test, LGBM_PARAMS)

    # 4. Evaluate
    base_feature_names = [c for c in list(dfs.values())[0].columns if c not in DROP_COLUMNS]
    evaluate(model, X_test, y_test, base_feature_names)

    # 5. Save
    save_model(model, MODEL_OUT)

    print("\nDone! Next steps:")
    print("  - Review confusion_matrix.png to see where errors occur")
    print("  - Review feature_importance.png to understand key features")
    print("  - Use predict_on_new_csv() for inference on new recordings")