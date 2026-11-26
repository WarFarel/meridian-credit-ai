import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_SEED = 42
TARGET_COL = "default_within_24m"
ID_COL = "application_id"

# Protected attributes separated so they are not used directly in scoring,
# but kept alongside the test set for post-hoc fairness checks.
PROTECTED_COLS = ["sex", "marital_status", "age_years"]

# Two genuinely ordinal variables ordered from lowest to highest stability/standing
ORDINAL_COLS = ["residency_status", "credit_history"]
ORDINAL_CATEGORIES = [
    ["Visa Holder", "Settled Status", "UK Citizen"],
    [
        "Critical account",
        "Past delays",
        "No credits taken",
        "Existing credits paid",
        "All paid on time",
    ],
]

# Checking account status has an ordered balance scale, except 'No account' sits outside it
OUT_OF_SCALE_COL = "checking_account_status"
OUT_OF_SCALE_VALID = ["Overdrawn", "0 to 500 GBP", "500 GBP or more"]
OUT_OF_SCALE_SPECIAL = "No account"

NOMINAL_COLS = [
    "region",
    "employment_status",
    "housing_status",
    "loan_purpose",
    "application_channel",
]

BINARY_YES_NO_COLS = [
    "has_savings_account",
    "has_guarantor",
    "previous_application_declined",
]

NUMERIC_COLS = [
    "num_dependents",
    "months_in_current_job",       # Structural 0 when unemployed/retired (kept as 0)
    "annual_income_gbp",
    "monthly_expenses_gbp",
    "years_at_current_address",    # ~2.8% missing -> median imputed + missing indicator
    "existing_debt_gbp",
    "num_existing_credit_lines",
    "num_late_payments_24m",
    "savings_balance_gbp",         # ~4.3% missing -> median imputed + missing indicator
    "loan_amount_gbp",
    "loan_term_months",
]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Creates debt-to-income/affordability ratios and separates out-of-scale categories."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert Yes/No text flags to 1/0
        for col in BINARY_YES_NO_COLS:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "1": 1, "no": 0, "0": 0})
            )

        annual_inc = np.maximum(df["annual_income_gbp"], 1.0)
        monthly_inc = np.maximum(df["annual_income_gbp"] / 12.0, 1.0)
        term_months = np.maximum(df["loan_term_months"], 1.0)
        monthly_repayment = df["loan_amount_gbp"] / term_months

        # Domain ratios required for credit affordability assessment
        df["debt_to_income_ratio"] = df["existing_debt_gbp"] / annual_inc
        df["loan_to_income_ratio"] = df["loan_amount_gbp"] / annual_inc
        df["expense_to_income_ratio"] = (df["monthly_expenses_gbp"] * 12.0) / annual_inc
        df["affordability_ratio"] = (
            df["monthly_expenses_gbp"] + monthly_repayment
        ) / monthly_inc
        df["net_monthly_disposable_gbp"] = (
            monthly_inc - df["monthly_expenses_gbp"] - monthly_repayment
        )

        # Isolate 'No account' into its own binary flag so it isn't ranked on the balance scale
        df["has_no_checking_account"] = (
            df[OUT_OF_SCALE_COL] == OUT_OF_SCALE_SPECIAL
        ).astype(int)
        df["clean_checking_tier"] = df[OUT_OF_SCALE_COL].where(
            df[OUT_OF_SCALE_COL].isin(OUT_OF_SCALE_VALID), other=np.nan
        )

        # Separate indicator for thin credit files ('No credits taken')
        df["is_thin_credit_file"] = (
            df["credit_history"] == "No credits taken"
        ).astype(int)

        return df


def make_preprocessing_pipeline(scale_numeric: bool = False) -> Pipeline:
    """
    Returns an unfitted preprocessing pipeline to prevent data leakage.
    Set scale_numeric=True for linear models, False for tree-based models.
    """
    num_features = (
        NUMERIC_COLS
        + BINARY_YES_NO_COLS
        + [
            "debt_to_income_ratio",
            "loan_to_income_ratio",
            "expense_to_income_ratio",
            "affordability_ratio",
            "net_monthly_disposable_gbp",
            "has_no_checking_account",
            "is_thin_credit_file",
        ]
    )

    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_features),
            (
                "ord",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=ORDINAL_CATEGORIES,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),
                ORDINAL_COLS,
            ),
            (
                "oos",
                Pipeline([
                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[OUT_OF_SCALE_VALID],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    )
                ]),
                ["clean_checking_tier"],
            ),
            (
                "nom",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                NOMINAL_COLS,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("features", FeatureBuilder()),
        ("encode", col_transformer),
    ])


def load_dataset(csv_path: str = "data/creditsense_applications.csv"):
    """Loads raw CSV, drops application_id, and returns a stratified 80/20 split."""
    df = pd.read_csv(csv_path)

    protected = df[PROTECTED_COLS].copy()
    protected["age_band"] = pd.cut(
        protected["age_years"],
        bins=[17, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "51+"],
    )

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[ID_COL, TARGET_COL] + PROTECTED_COLS)

    return train_test_split(
        X, y, protected, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )