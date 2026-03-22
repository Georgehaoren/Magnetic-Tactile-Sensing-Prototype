import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_artifacts(
    feature_csv: Path,
    model_path: Path,
    metrics_path: Path | None = None,
) -> tuple[pd.DataFrame, object, dict | None]:
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    feat_df = pd.read_csv(feature_csv)
    model_pipe = joblib.load(model_path)

    metrics = None
    if metrics_path is not None and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return feat_df, model_pipe, metrics


def infer_task_name(model_path: Path) -> str:
    name = model_path.stem
    if "contact_vs_no_contact" in name:
        return "contact_vs_no_contact"
    if "touch_vs_punch" in name:
        return "touch_vs_punch"
    return name


def prepare_task_dataframe(feat_df: pd.DataFrame, task_name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = feat_df.copy()

    if task_name == "contact_vs_no_contact":
        y = df["label"].apply(lambda x: "contact" if x in {"touch", "punch"} else "no_contact")
    elif task_name == "touch_vs_punch":
        df = df[df["label"].isin(["touch", "punch"])].copy()
        y = df["label"]
    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    non_feature_cols = {"segment_id", "label", "task_label"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].copy()

    return X, y


def get_preprocessed_X_and_feature_names(model_pipe, X_raw: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    preprocessor = model_pipe.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_raw)

    # ColumnTransformer + numeric pipeline; order should match raw columns
    feature_names = list(X_raw.columns)

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    return X_proc, feature_names


def get_base_model(model_pipe):
    return model_pipe.named_steps["model"]


def make_output_dir(base_outdir: Path, task_name: str, model_name: str) -> Path:
    outdir = base_outdir / f"{task_name}_{model_name}_shap"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_top_features(shap_values_abs_mean: np.ndarray, feature_names: List[str], out_csv: Path, top_k: int = 20):
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_values_abs_mean
    }).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv(out_csv, index=False)
    print("\nTop SHAP features:")
    print(imp_df.head(top_k).to_string(index=False))
    return imp_df


def plot_summary_bar(shap_values, X_proc_df: pd.DataFrame, out_path: Path, max_display: int = 15):
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_proc_df,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_beeswarm(shap_values, X_proc_df: pd.DataFrame, out_path: Path, max_display: int = 15):
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_proc_df,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_dependence_for_top_features(shap_values, X_proc_df: pd.DataFrame, top_features: List[str], out_dir: Path):
    for feat in top_features:
        plt.figure()
        shap.dependence_plot(
            feat,
            shap_values,
            X_proc_df,
            show=False
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"dependence_{feat}.png", dpi=200, bbox_inches="tight")
        plt.close()


def plot_waterfall_for_sample(explanation, sample_idx: int, out_path: Path, max_display: int = 15):
    plt.figure()
    shap.plots.waterfall(explanation[sample_idx], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def explain_logistic_regression(model_pipe, X_raw: pd.DataFrame, outdir: Path, sample_idx: int = 0):
    X_proc, feature_names = get_preprocessed_X_and_feature_names(model_pipe, X_raw)
    model = get_base_model(model_pipe)

    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

    explainer = shap.LinearExplainer(model, X_proc_df, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_proc_df)

    # binary logistic regression often returns shape (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    abs_mean = np.abs(shap_vals).mean(axis=0)
    imp_df = save_top_features(abs_mean, feature_names, outdir / "top_features.csv")

    plot_summary_bar(shap_vals, X_proc_df, outdir / "summary_bar.png")
    plot_summary_beeswarm(shap_vals, X_proc_df, outdir / "summary_beeswarm.png")

    top_feats = imp_df["feature"].head(5).tolist()
    plot_dependence_for_top_features(shap_vals, X_proc_df, top_feats, outdir)

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=np.repeat(explainer.expected_value, len(X_proc_df)),
        data=X_proc_df.values,
        feature_names=feature_names,
    )

    sample_idx = min(sample_idx, len(X_proc_df) - 1)
    plot_waterfall_for_sample(explanation, sample_idx, outdir / f"waterfall_sample_{sample_idx}.png")

    return imp_df


def explain_random_forest(model_pipe, X_raw: pd.DataFrame, outdir: Path, sample_idx: int = 0):
    X_proc, feature_names = get_preprocessed_X_and_feature_names(model_pipe, X_raw)
    model = get_base_model(model_pipe)

    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc_df)

    # binary RF may return list of 2 arrays or a 3D array depending on version
    if isinstance(shap_values, list):
        # use positive class if available
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    else:
        # if shape is (n_samples, n_features, n_classes) or similar, handle lightly
        if shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
            base_value = explainer.expected_value[1] if np.ndim(explainer.expected_value) > 0 else explainer.expected_value
        else:
            shap_vals = shap_values
            base_value = explainer.expected_value

    abs_mean = np.abs(shap_vals).mean(axis=0)
    imp_df = save_top_features(abs_mean, feature_names, outdir / "top_features.csv")

    plot_summary_bar(shap_vals, X_proc_df, outdir / "summary_bar.png")
    plot_summary_beeswarm(shap_vals, X_proc_df, outdir / "summary_beeswarm.png")

    top_feats = imp_df["feature"].head(5).tolist()
    plot_dependence_for_top_features(shap_vals, X_proc_df, top_feats, outdir)

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=np.repeat(base_value, len(X_proc_df)),
        data=X_proc_df.values,
        feature_names=feature_names,
    )

    sample_idx = min(sample_idx, len(X_proc_df) - 1)
    plot_waterfall_for_sample(explanation, sample_idx, outdir / f"waterfall_sample_{sample_idx}.png")

    return imp_df


def main():
    parser = argparse.ArgumentParser(description="Run SHAP explainability on trained tactile sensing baselines.")
    parser.add_argument("--features", type=str, required=True, help="Path to segment_features.csv")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .joblib model")
    parser.add_argument("--metrics", type=str, default=None, help="Optional metrics JSON path")
    parser.add_argument("--outdir", type=str, default="explanations", help="Output directory")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index for waterfall plot")
    args = parser.parse_args()

    feature_csv = Path(args.features)
    model_path = Path(args.model)
    metrics_path = Path(args.metrics) if args.metrics else None
    base_outdir = Path(args.outdir)

    feat_df, model_pipe, metrics = load_artifacts(feature_csv, model_path, metrics_path)
    task_name = infer_task_name(model_path)
    model_name = model_path.stem.split("_")[-1]

    X_raw, y = prepare_task_dataframe(feat_df, task_name)

    outdir = make_output_dir(base_outdir, task_name, model_name)
    print(f"Task: {task_name}")
    print(f"Output dir: {outdir}")

    base_model = get_base_model(model_pipe)
    model_class_name = base_model.__class__.__name__.lower()

    if "logisticregression" in model_class_name:
        explain_logistic_regression(model_pipe, X_raw, outdir, sample_idx=args.sample_idx)
    elif "randomforestclassifier" in model_class_name:
        explain_random_forest(model_pipe, X_raw, outdir, sample_idx=args.sample_idx)
    else:
        raise ValueError(f"Unsupported model class for SHAP script: {base_model.__class__.__name__}")

    print("\nSaved SHAP outputs:")
    for p in sorted(outdir.glob("*")):
        print(f"- {p}")


if __name__ == "__main__":
    main()