import json
import gradio as gr
import joblib
import pandas as pd

# Load trained model, SHAP explainer, and operating threshold
model = joblib.load("model_artifacts/credit_model.joblib")
explainer = joblib.load("model_artifacts/shap_explainer.joblib")
with open("model_artifacts/metadata.json") as f:
    meta = json.load(f)

THRESHOLD = meta["threshold"]
FEATURE_NAMES = meta["feature_names"]

# Plain-English translations for adverse-action notices
REASON_TEXT = {
    "debt_to_income_ratio": "Your existing debt level is high compared to your annual income.",
    "existing_debt_gbp": "Your total existing debt balance exceeds our lending risk threshold.",
    "affordability_ratio": "Monthly living expenses and loan repayments represent too high a share of your income.",
    "net_monthly_disposable_gbp": "Your remaining disposable income after regular expenses is below our minimum buffer.",
    "expense_to_income_ratio": "Your regular living expenses are high relative to your income.",
    "loan_to_income_ratio": "The requested loan amount is large compared to your annual income.",
    "loan_amount_gbp": "The requested loan amount increases repayment risk given your profile.",
    "num_late_payments_24m": "Your credit file shows recent late payments within the past 24 months.",
    "credit_history": "Your overall credit repayment history does not meet our current lending criteria.",
    "clean_checking_tier": "Your current account balance tier indicates limited short-term cash buffer.",
    "has_no_checking_account": "We could not verify an active UK current account to service repayments.",
    "savings_balance_gbp": "Your savings balance provides a limited cushion for unexpected costs.",
    "months_in_current_job": "Your time in your current job is shorter than our employment stability benchmark.",
    "years_at_current_address": "Your time at your current address is shorter than our stability benchmark.",
    "previous_application_declined": "A recent previous credit application decline indicates elevated borrowing risk.",
    "is_thin_credit_file": "There is limited prior borrowing history on your credit file.",
}


def assess_application(
    residency_status, num_dependents, region, employment_status,
    months_in_current_job, annual_income_gbp, monthly_expenses_gbp,
    housing_status, years_at_current_address, existing_debt_gbp,
    num_existing_credit_lines, num_late_payments_24m, credit_history,
    has_savings_account, savings_balance_gbp, checking_account_status,
    loan_amount_gbp, loan_term_months, loan_purpose, has_guarantor,
    previous_application_declined, application_channel,
):
    row = pd.DataFrame([{
        "residency_status": residency_status,
        "num_dependents": int(num_dependents),
        "region": region,
        "employment_status": employment_status,
        "months_in_current_job": int(months_in_current_job),
        "annual_income_gbp": float(annual_income_gbp),
        "monthly_expenses_gbp": float(monthly_expenses_gbp),
        "housing_status": housing_status,
        "years_at_current_address": float(years_at_current_address),
        "existing_debt_gbp": float(existing_debt_gbp),
        "num_existing_credit_lines": int(num_existing_credit_lines),
        "num_late_payments_24m": int(num_late_payments_24m),
        "credit_history": credit_history,
        "has_savings_account": has_savings_account,
        "savings_balance_gbp": float(savings_balance_gbp),
        "checking_account_status": checking_account_status,
        "loan_amount_gbp": float(loan_amount_gbp),
        "loan_term_months": int(loan_term_months),
        "loan_purpose": loan_purpose,
        "has_guarantor": has_guarantor,
        "previous_application_declined": previous_application_declined,
        "application_channel": application_channel,
    }])

    prob = float(model.predict_proba(row)[0, 1])
    declined = prob >= THRESHOLD

    # Technical SHAP explanation
    transformed = model.named_steps["prep"].transform(row)
    shap_vals = explainer.shap_values(transformed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    ranked = sorted(zip(FEATURE_NAMES, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)

    tech_df = pd.DataFrame([
        {
            "Feature": feat,
            "SHAP Impact": round(float(val), 4),
            "Direction": "Increases Risk" if val > 0 else "Reduces Risk",
        }
        for feat, val in ranked[:6]
    ])

    # Plain-English applicant notice
    if not declined:
        notice = (
            "APPLICATION OUTCOME: APPROVED\n\n"
            "Your application meets Meridian Credit's affordability and creditworthiness criteria."
        )
    else:
        reasons = []
        for feat, val in ranked:
            if val <= 0:
                continue
            clean_key = feat.split("__")[-1]
            if clean_key in REASON_TEXT and REASON_TEXT[clean_key] not in reasons:
                reasons.append(REASON_TEXT[clean_key])
            elif clean_key.startswith("employment_status_"):
                msg = "Your current employment arrangement indicates variable or limited income continuity."
                if msg not in reasons:
                    reasons.append(msg)
            if len(reasons) == 3:
                break

        if not reasons:
            reasons.append("Your overall financial profile did not meet our minimum risk threshold.")

        bullet_list = "\n".join(f"- {r}" for r in reasons)
        notice = (
            "NOTICE OF CREDIT DECISION — APPLICATION DECLINED\n\n"
            "Thank you for applying to Meridian Credit. After assessing your application, "
            "we are unable to approve your loan request at this time.\n\n"
            "Main reasons for this decision:\n"
            f"{bullet_list}\n\n"
            "Your rights and next steps:\n"
            "- You have the right to request a human review of this automated decision by a "
            "Meridian Credit underwriter within 30 days.\n"
            "- Applying for a smaller loan amount or reducing existing debt may improve eligibility."
        )

    banner = f"{'DECLINED' if declined else 'APPROVED'} | Predicted Default Risk: {prob:.1%} (Cutoff: {THRESHOLD:.1%})"
    return banner, tech_df, notice


with gr.Blocks(title="Meridian Credit Underwriting Prototype") as demo:
    gr.Markdown("# Meridian Credit — Explainable Credit Risk System")
    gr.Markdown("Generates internal SHAP attributions for underwriters and plain-English notices for applicants.")

    with gr.Row():
        with gr.Column():
            residency = gr.Dropdown(["UK Citizen", "Settled Status", "Visa Holder"], value="UK Citizen", label="Residency Status")
            emp_status = gr.Dropdown(["Permanent", "Self-employed", "Fixed-term", "Zero-hours", "Unemployed", "Retired"], value="Permanent", label="Employment Status")
            job_months = gr.Number(value=36, label="Months in Current Job")
            income = gr.Number(value=38000, label="Annual Income (£)")
            expenses = gr.Number(value=1200, label="Monthly Expenses (£)")
            housing = gr.Dropdown(["Renting", "Mortgage", "Living with family", "Owner"], value="Mortgage", label="Housing Status")
            addr_years = gr.Number(value=4.0, label="Years at Current Address")
            dependents = gr.Number(value=1, label="Dependents")
            region = gr.Dropdown(["London", "South East", "Midlands", "North West", "Yorkshire", "Scotland", "South West", "North East", "Wales", "Northern Ireland"], value="Midlands", label="Region")

        with gr.Column():
            existing_debt = gr.Number(value=3500, label="Existing Debt (£)")
            credit_lines = gr.Number(value=3, label="Existing Credit Lines")
            late_payments = gr.Number(value=0, label="Late Payments (24m)")
            cred_hist = gr.Dropdown(["All paid on time", "Existing credits paid", "No credits taken", "Past delays", "Critical account"], value="All paid on time", label="Credit History")
            has_sav = gr.Dropdown(["Yes", "No"], value="Yes", label="Has Savings Account")
            sav_bal = gr.Number(value=4500, label="Savings Balance (£)")
            chk_stat = gr.Dropdown(["500 GBP or more", "0 to 500 GBP", "Overdrawn", "No account"], value="500 GBP or more", label="Checking Account Status")
            loan_amt = gr.Number(value=8000, label="Requested Loan (£)")
            loan_term = gr.Number(value=36, label="Loan Term (Months)")
            purpose = gr.Dropdown(["Car", "Debt consolidation", "Home improvement", "Education", "Other", "Business", "Holiday", "Medical"], value="Home improvement", label="Loan Purpose")
            guarantor = gr.Dropdown(["Yes", "No"], value="No", label="Has Guarantor")
            prev_dec = gr.Dropdown(["Yes", "No"], value="No", label="Previous Application Declined")
            channel = gr.Dropdown(["Online", "Broker", "Branch", "Phone"], value="Online", label="Application Channel")

    run_btn = gr.Button("Assess Application", variant="primary")
    outcome_box = gr.Textbox(label="Decision Summary")

    with gr.Row():
        tech_out = gr.Dataframe(label="Technical Explanation (Top SHAP Features)")
        notice_out = gr.Textbox(label="Plain-English Applicant Notice", lines=12)

    inputs = [
        residency, dependents, region, emp_status, job_months, income,
        expenses, housing, addr_years, existing_debt, credit_lines,
        late_payments, cred_hist, has_sav, sav_bal, chk_stat, loan_amt,
        loan_term, purpose, guarantor, prev_dec, channel,
    ]
    run_btn.click(fn=assess_application, inputs=inputs, outputs=[outcome_box, tech_out, notice_out])

    gr.Examples(
        label="Demo Cases (Approved, Declined, and Test-Set Failure Case)",
        inputs=inputs,
        examples=[
            ["UK Citizen", 1, "Midlands", "Permanent", 48, 42000, 1100, "Mortgage", 5.0, 2000, 3, 0, "All paid on time", "Yes", 6000, "500 GBP or more", 7500, 36, "Car", "No", "No", "Online"],
            ["Visa Holder", 2, "London", "Zero-hours", 4, 21000, 1450, "Renting", 0.5, 14000, 5, 3, "Past delays", "No", 0, "Overdrawn", 12000, 24, "Debt consolidation", "No", "Yes", "Broker"],
            ["UK Citizen", 0, "London", "Permanent", 34, 116480, 4570, "Mortgage", 8.5, 0, 3, 0, "Existing credits paid", "No", 0, "0 to 500 GBP", 19800, 48, "Business", "No", "No", "Online"],
        ],
    )

if __name__ == "__main__":
    demo.launch()