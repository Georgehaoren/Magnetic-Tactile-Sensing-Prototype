import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = {"no_contact", "touch", "punch"}


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"timestamp", "gx", "gy", "gz", "B", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df = df[df["label"].isin(LABELS)].copy()
    if df.empty:
        raise ValueError("No valid labeled rows found.")

    for col in ["timestamp", "gx", "gy", "gz", "B"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "gx", "gy", "gz", "B", "label"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def assign_segments(
    df: pd.DataFrame,
    gap_threshold_sec: float = 0.35,
    min_segment_samples: int = 8,
) -> pd.DataFrame:
    """
    Split continuous labeled stream into segments.
    A new segment starts when:
      - label changes, or
      - time gap exceeds threshold
    """
    df = df.copy()
    df["dt"] = df["timestamp"].diff().fillna(0.0)

    new_segment = (
        (df["label"] != df["label"].shift(1)) |
        (df["dt"] > gap_threshold_sec)
    )

    df["segment_id"] = new_segment.cumsum()

    # remove tiny segments
    segment_sizes = df.groupby("segment_id").size()
    valid_segments = segment_sizes[segment_sizes >= min_segment_samples].index
    df = df[df["segment_id"].isin(valid_segments)].copy()

    # reindex segment ids to be clean
    seg_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df["segment_id"].unique()))}
    df["segment_id"] = df["segment_id"].map(seg_map)

    return df.reset_index(drop=True)


def safe_rise_time(signal: np.ndarray, time_arr: np.ndarray) -> float:
    """
    Approximate rise time from 10% to 90% of signal excursion.
    """
    if len(signal) < 3:
        return np.nan

    s0 = np.median(signal[: max(1, len(signal) // 10)])
    smax = np.max(signal)
    amp = smax - s0
    if abs(amp) < 1e-9:
        return np.nan

    low = s0 + 0.1 * amp
    high = s0 + 0.9 * amp

    try:
        t_low = time_arr[np.where(signal >= low)[0][0]]
        t_high = time_arr[np.where(signal >= high)[0][0]]
        return float(t_high - t_low)
    except Exception:
        return np.nan


def zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    signs = np.sign(x)
    return float(np.mean(signs[1:] != signs[:-1]))


def extract_segment_features(seg: pd.DataFrame) -> Dict:
    seg = seg.sort_values("timestamp").reset_index(drop=True)

    t = seg["timestamp"].to_numpy()
    t_rel = t - t[0]

    gx = seg["gx"].to_numpy(dtype=float)
    gy = seg["gy"].to_numpy(dtype=float)
    gz = seg["gz"].to_numpy(dtype=float)
    B = seg["B"].to_numpy(dtype=float)

    # baseline = first 20% of the segment, at least 3 samples
    n = len(seg)
    base_n = max(3, int(0.2 * n))
    base_slice = slice(0, base_n)

    gx0 = float(np.mean(gx[base_slice]))
    gy0 = float(np.mean(gy[base_slice]))
    gz0 = float(np.mean(gz[base_slice]))
    B0 = float(np.mean(B[base_slice]))

    dgx = gx - gx0
    dgy = gy - gy0
    dgz = gz - gz0
    dB = B - B0

    dt = np.diff(t_rel)
    dt[dt == 0] = 1e-6

    dB_dt = np.diff(B) / dt if len(B) > 1 else np.array([0.0])
    dgx_dt = np.diff(gx) / dt if len(gx) > 1 else np.array([0.0])
    dgy_dt = np.diff(gy) / dt if len(gy) > 1 else np.array([0.0])
    dgz_dt = np.diff(gz) / dt if len(gz) > 1 else np.array([0.0])

    # hold region = middle 40%
    start_hold = int(0.3 * n)
    end_hold = max(start_hold + 1, int(0.7 * n))
    hold_B = B[start_hold:end_hold]
    hold_dB = dB[start_hold:end_hold]

    features = {
        "segment_id": int(seg["segment_id"].iloc[0]),
        "label": seg["label"].iloc[0],
        "n_samples": int(n),
        "duration_sec": float(t_rel[-1]) if n > 1 else 0.0,

        # raw level features
        "gx_mean": float(np.mean(gx)),
        "gy_mean": float(np.mean(gy)),
        "gz_mean": float(np.mean(gz)),
        "B_mean": float(np.mean(B)),

        "gx_std": float(np.std(gx)),
        "gy_std": float(np.std(gy)),
        "gz_std": float(np.std(gz)),
        "B_std": float(np.std(B)),

        # baseline-relative features
        "dgx_mean": float(np.mean(dgx)),
        "dgy_mean": float(np.mean(dgy)),
        "dgz_mean": float(np.mean(dgz)),
        "dB_mean": float(np.mean(dB)),

        "dgx_max_abs": float(np.max(np.abs(dgx))),
        "dgy_max_abs": float(np.max(np.abs(dgy))),
        "dgz_max_abs": float(np.max(np.abs(dgz))),
        "dB_max_abs": float(np.max(np.abs(dB))),

        "gx_range": float(np.max(gx) - np.min(gx)),
        "gy_range": float(np.max(gy) - np.min(gy)),
        "gz_range": float(np.max(gz) - np.min(gz)),
        "B_range": float(np.max(B) - np.min(B)),

        # dynamic features
        "dB_dt_max": float(np.max(dB_dt)) if len(dB_dt) else 0.0,
        "dB_dt_min": float(np.min(dB_dt)) if len(dB_dt) else 0.0,
        "dB_dt_max_abs": float(np.max(np.abs(dB_dt))) if len(dB_dt) else 0.0,

        "dgx_dt_max_abs": float(np.max(np.abs(dgx_dt))) if len(dgx_dt) else 0.0,
        "dgy_dt_max_abs": float(np.max(np.abs(dgy_dt))) if len(dgy_dt) else 0.0,
        "dgz_dt_max_abs": float(np.max(np.abs(dgz_dt))) if len(dgz_dt) else 0.0,

        "rise_time_B": safe_rise_time(B, t_rel),
        "rise_time_dB": safe_rise_time(dB, t_rel),

        # hold stability
        "hold_B_mean": float(np.mean(hold_B)) if len(hold_B) else np.nan,
        "hold_B_std": float(np.std(hold_B)) if len(hold_B) else np.nan,
        "hold_dB_mean": float(np.mean(hold_dB)) if len(hold_dB) else np.nan,
        "hold_dB_std": float(np.std(hold_dB)) if len(hold_dB) else np.nan,

        # shape / simple signal character
        "B_peak_idx_ratio": float(np.argmax(B) / max(1, n - 1)),
        "dB_peak_idx_ratio": float(np.argmax(np.abs(dB)) / max(1, n - 1)),
        "B_zero_cross_rate": float(zero_crossing_rate(dB)),
        "B_energy": float(np.mean(B ** 2)),
        "dB_energy": float(np.mean(dB ** 2)),

        # final values
        "gx_last": float(gx[-1]),
        "gy_last": float(gy[-1]),
        "gz_last": float(gz[-1]),
        "B_last": float(B[-1]),
        "dB_last": float(dB[-1]),
    }

    return features


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []
    for _, seg in df.groupby("segment_id"):
        feature_rows.append(extract_segment_features(seg))
    feat_df = pd.DataFrame(feature_rows)
    return feat_df


def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_cols),
        ]
    )
    return preprocessor


def train_and_evaluate(
    feat_df: pd.DataFrame,
    label_col: str,
    positive_task_name: str,
    out_dir: Path,
    model_name: str = "logreg",
    test_size: float = 0.3,
    random_state: int = 42,
):
    non_feature_cols = {"segment_id", "label", "task_label"}
    feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]

    X = feat_df[feature_cols].copy()
    y = feat_df[label_col].copy()

    unique_y = sorted(pd.unique(y))
    if len(unique_y) < 2:
        raise ValueError(f"Task '{positive_task_name}' has fewer than 2 classes: {unique_y}")

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = make_preprocessor(feature_cols)

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError("model_name must be 'logreg' or 'rf'")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 80}")
    print(f"Task: {positive_task_name}")
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "task": positive_task_name,
        "model": model_name,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_cols": feature_cols,
    }

    # feature importance
    feature_importance_df = None
    model = pipe.named_steps["model"]

    if model_name == "logreg":
        coef = model.coef_
        if coef.ndim == 2 and coef.shape[0] == 1:
            importance = np.abs(coef[0])
        else:
            importance = np.mean(np.abs(coef), axis=0)

        feature_importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance}
        ).sort_values("importance", ascending=False)

    elif model_name == "rf":
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance}
        ).sort_values("importance", ascending=False)

    # save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{positive_task_name}_{model_name}.joblib"
    metrics_path = out_dir / f"{positive_task_name}_{model_name}_metrics.json"
    feature_imp_path = out_dir / f"{positive_task_name}_{model_name}_feature_importance.csv"

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if feature_importance_df is not None:
        feature_importance_df.to_csv(feature_imp_path, index=False)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    if feature_importance_df is not None:
        print(f"Saved feature importance to: {feature_imp_path}")
        print("\nTop 10 features:")
        print(feature_importance_df.head(10).to_string(index=False))

    return pipe, metrics, feature_importance_df


def build_contact_task(feat_df: pd.DataFrame) -> pd.DataFrame:
    df = feat_df.copy()
    df["task_label"] = df["label"].apply(lambda x: "contact" if x in {"touch", "punch"} else "no_contact")
    return df


def build_touch_punch_task(feat_df: pd.DataFrame) -> pd.DataFrame:
    df = feat_df[feat_df["label"].isin(["touch", "punch"])].copy()
    df["task_label"] = df["label"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Train tactile sensing baselines from labeled CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to labeled CSV from Streamlit app.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save artifacts.")
    parser.add_argument("--gap", type=float, default=0.35, help="Max time gap within one segment (sec).")
    parser.add_argument("--min_samples", type=int, default=8, help="Minimum samples per segment.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test split ratio.")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"], help="Baseline model.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.outdir)

    print("Loading CSV...")
    df = load_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    print("Assigning segments...")
    seg_df = assign_segments(df, gap_threshold_sec=args.gap, min_segment_samples=args.min_samples)
    print(f"Remaining rows after segmentation filter: {len(seg_df)}")
    print(f"Number of segments: {seg_df['segment_id'].nunique()}")

    if seg_df.empty:
        raise ValueError("No valid segments after filtering.")

    print("Building feature table...")
    feat_df = build_feature_table(seg_df)
    print(f"Feature table shape: {feat_df.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "segment_features.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Saved segment features to: {feat_path}")

    print("\nSegment counts by label:")
    print(feat_df["label"].value_counts())

    # Task 1: contact vs no_contact
    contact_df = build_contact_task(feat_df)
    train_and_evaluate(
        contact_df,
        label_col="task_label",
        positive_task_name="contact_vs_no_contact",
        out_dir=out_dir,
        model_name=args.model,
        test_size=args.test_size,
    )

    # Task 2: touch vs punch
    tp_df = build_touch_punch_task(feat_df)
    if tp_df["task_label"].nunique() >= 2:
        train_and_evaluate(
            tp_df,
            label_col="task_label",
            positive_task_name="touch_vs_punch",
            out_dir=out_dir,
            model_name=args.model,
            test_size=args.test_size,
        )
    else:
        print("\nSkipping touch_vs_punch: not enough class diversity.")

    print("\nDone.")


if __name__ == "__main__":
    main()