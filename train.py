

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    FeatureEngineer,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PIPELINE_PATH = BASE_DIR / "pipeline.pkl"
PLOT_DIR = BASE_DIR / "static"
PLOT_PATH = PLOT_DIR / "feature_importance.png"

RANDOM_STATE = 42


def _get_candidate_models(y_train):
    """Return a dict of model_name -> model_instance to compare, handling class imbalance."""
    # Calculate pos_weight for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        ),
    }

    # XGBoost (optional dependency)
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            scale_pos_weight=scale_pos_weight,
        )
    except ImportError:
        print("  [!] xgboost not installed — skipping XGBoost.")

    # CatBoost (optional dependency)
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=200, random_state=RANDOM_STATE, verbose=0, auto_class_weights="Balanced"
        )
    except ImportError:
        print("  [!] catboost not installed — skipping CatBoost.")

    return models


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute and return comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)

    # Probability estimates for ROC-AUC (handle models without predict_proba)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        roc_auc = None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc,
        "CV-ROC-AUC": None,  # Will be filled during training
    }

    return metrics, y_pred


def print_detailed_report(y_test, y_pred, model_name):
    """Print confusion matrix and classification report."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Detailed Report")
    print(f"{'=' * 60}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0][0]:>5}   FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}   TP={cm[1][1]:>5}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))


def plot_feature_importance(pipeline, top_n=20):
    """
    Extract feature names from the pipeline's ColumnTransformer, get
    importances from the model, and save a horizontal bar chart.
    """
    # Extract components from the pipeline
    column_transformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Build full feature name list from the ColumnTransformer
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if name == "cat":
            # Get OHE feature names
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == "num":
            feature_names.extend(columns)

    # Get importances based on model type
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        importance_label = "Absolute Coefficient Value"
    else:
        print("  [!] Model does not expose feature importances — skipping plot.")
        return

    
    importance_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances,
    })
    importance_df = importance_df.sort_values(
        "Importance", ascending=True
    ).tail(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color="#2196F3",
        edgecolor="#1565C0",
    )
    ax.set_xlabel(importance_label, fontsize=12)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Feature importance plot saved to: {PLOT_PATH}")


def main():
    print("=" * 60)
    print("  Employee Attrition Prediction - Training Pipeline")
    print("=" * 60)

    
    if not DATA_PATH.exists():
        print(f"\n  [ERROR] Dataset not found at: {DATA_PATH}")
        print("  Please place 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in the data/ folder.")
        return
    
    print(f"\n  [1/6] Loading dataset from {DATA_PATH.name} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"        Dataset shape: {df.shape}")

    
    y = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
    X = df.drop(columns=["Attrition"])
    print(f"        Target distribution: {dict(y.value_counts())}")

    
    print("\n  [2/6] Splitting data (80% train / 20% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"        Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    
    print("\n  [3/6] Building preprocessing pipeline ...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    
    print("\n  [4/7] Comparing models via 5-fold Stratified CV (training data only) ...\n")
    candidate_models = _get_candidate_models(y_train)
    cv_results = []  # list of {"Model": name, "CV-ROC-AUC": mean_score}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in candidate_models.items():
        print(f"    Cross-validating {name} ...")
        pipeline = Pipeline([
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        cv_results.append({"Model": name, "CV-ROC-AUC": cv_mean})
        print(f"      → CV ROC-AUC: {cv_mean:.4f} (±{cv_std:.4f})")

    # --- CV Comparison Table ---
    print("\n" + "=" * 60)
    print("  Model Comparison (CV ROC-AUC on Training Data Only)")
    print("=" * 60)
    comparison_df = pd.DataFrame(cv_results).set_index("Model")
    comparison_df["CV-ROC-AUC"] = comparison_df["CV-ROC-AUC"].apply(lambda x: f"{x:.4f}")
    print(f"\n{comparison_df.to_string()}\n")

    # --- Select best model by mean CV ROC-AUC ---
    print("  [5/7] Selecting best model (by mean CV ROC-AUC) ...")
    best_entry = max(cv_results, key=lambda m: m["CV-ROC-AUC"])
    best_name = best_entry["Model"]
    print(f"        Best model: {best_name} (CV ROC-AUC = {best_entry['CV-ROC-AUC']:.4f})")

    # Rebuild the best pipeline and fit on full training data
    best_model = candidate_models[best_name]
    best_pipeline = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CATEGORICAL_FEATURES),
                ("num", "passthrough", NUMERICAL_FEATURES),
            ]
        )),
        ("model", best_model),
    ])
    best_pipeline.fit(X_train, y_train)

    # --- Single final evaluation on the untouched test set ---
    print(f"\n  [6/7] Evaluating {best_name} on the held-out test set ...")
    metrics, y_pred = evaluate_model(best_pipeline, X_test, y_test, best_name)
    print_detailed_report(y_test, y_pred, best_name)

    print("\n" + "=" * 60)
    print(f"  Final Test-Set Performance — {best_name}")
    print("=" * 60)
    for metric_name in ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"):
        val = metrics[metric_name]
        print(f"    {metric_name:12s}: {val:.4f}" if val is not None else f"    {metric_name:12s}: N/A")

    plot_feature_importance(best_pipeline, top_n=20)

    print(f"\n  [7/7] Saving pipeline to {PIPELINE_PATH.name} ...")
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"        Pipeline saved successfully ({PIPELINE_PATH.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

