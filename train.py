import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import RANDOM_SEED, load_dataset, make_preprocessing_pipeline

# Declared design constraint: minimum 75% recall on defaults on validation data
RECALL_FLOOR = 0.75


def pick_threshold_for_recall(y_true, y_probs, target_recall: float) -> float:
    """Finds the highest probability cutoff that still achieves target_recall."""
    best_t = 0.50
    for t in np.linspace(0.05, 0.95, 181):
        preds = (y_probs >= t).astype(int)
        if recall_score(y_true, preds) >= target_recall:
            best_t = float(t)
    return best_t


def audit_fairness(prot_df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray):
    """Checks approval rates and wrongful-decline (FPR) rates across protected groups."""
    results = {}
    for col in ["sex", "marital_status", "age_band"]:
        group_rows = []
        for grp, idx in prot_df.groupby(col, observed=False).groups.items():
            pos = prot_df.index.get_indexer(idx)
            g_true = y_true.iloc[pos]
            g_pred = y_pred[pos]
            good_payers = g_true == 0
            fpr = float((g_pred[good_payers] == 1).mean()) if good_payers.sum() > 0 else np.nan
            group_rows.append({
                "group": str(grp),
                "count": int(len(pos)),
                "actual_default_pct": round(float(g_true.mean()) * 100, 1),
                "approval_rate_pct": round(float((g_pred == 0).mean()) * 100, 1),
                "wrongful_decline_fpr_pct": round(fpr * 100, 1),
            })
        results[col] = group_rows
    return results


def main():
    X_train, X_test, y_train, y_test, prot_train, prot_test = load_dataset()

    # Hold out a validation slice from X_train to tune threshold without touching X_test
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_SEED
    )

    # 1. Baseline comparison: Logistic Regression (needs scaled numeric features)
    lr_model = Pipeline([
        ("prep", make_preprocessing_pipeline(scale_numeric=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    lr_model.fit(X_tr, y_tr)
    lr_val_probs = lr_model.predict_proba(X_val)[:, 1]

    # 2. Primary model: LightGBM (tree splits are scale-invariant)
    lgb_model = Pipeline([
        ("prep", make_preprocessing_pipeline(scale_numeric=False)),
        ("clf", lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
        )),
    ])
    lgb_model.fit(X_tr, y_tr)
    lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]

    print("--- Validation Comparison ---")
    print(f"Logistic Regression | ROC-AUC: {roc_auc_score(y_val, lr_val_probs):.4f} | PR-AUC: {average_precision_score(y_val, lr_val_probs):.4f}")
    print(f"LightGBM            | ROC-AUC: {roc_auc_score(y_val, lgb_val_probs):.4f} | PR-AUC: {average_precision_score(y_val, lgb_val_probs):.4f}")

    threshold = pick_threshold_for_recall(y_val, lgb_val_probs, RECALL_FLOOR)

    # Retrain on full training set (4,800 rows) before single test evaluation
    lgb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    test_probs = lgb_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    print(f"\n--- Held-Out Test Set Results (Threshold = {threshold:.3f}) ---")
    print(f"Baseline LR ROC-AUC:   {roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]):.4f}")
    print(f"LightGBM ROC-AUC:      {roc_auc_score(y_test, test_probs):.4f}")
    print(f"LightGBM PR-AUC:       {average_precision_score(y_test, test_probs):.4f}")
    print(f"LightGBM Brier Score:  {brier_score_loss(y_test, test_probs):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=["Approve (0)", "Decline (1)"]))

    # Fairness audit
    fairness = audit_fairness(prot_test, y_test, test_preds)
    print("\n--- Fairness Audit ---")
    for attr, rows in fairness.items():
        print(f"\n[{attr}]")
        print(pd.DataFrame(rows).to_string(index=False))

    # Fit SHAP explainer and save artifacts for app.py
    feature_names = list(
        lgb_model.named_steps["prep"].named_steps["encode"].get_feature_names_out()
    )
    explainer = shap.TreeExplainer(lgb_model.named_steps["clf"])

    out_dir = Path("model_artifacts")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(lgb_model, out_dir / "credit_model.joblib")
    joblib.dump(explainer, out_dir / "shap_explainer.joblib")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "recall_floor": RECALL_FLOOR,
                "feature_names": feature_names,
                "fairness_audit": fairness,
            },
            f,
            indent=2,
        )
    print("\nSaved model artifacts to ./model_artifacts/")


if __name__ == "__main__":
    main()