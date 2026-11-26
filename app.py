import json
import gradio as gr
import joblib
import pandas as pd

# Load saved model, SHAP explainer, and threshold
model = joblib.load("model_artifacts/credit_model.joblib")
explainer = joblib.load("model_artifacts/shap_explainer.joblib")

with open("model_artifacts/metadata.json") as f:
    meta = json.load(f)

THRESHOLD = meta["threshold"]
FEATURE_NAMES = meta["feature_names"]

# Map technical feature names to plain-English reasons for the applicant
REASON_MAP = {
    "debt_to_income_ratio": "Existing debt is high relative to your annual income.",
    "existing_debt_gbp": "Total existing debt exceeds our lending threshold.",
    "affordability_ratio": "Monthly expenses plus loan repayments take up too much of your income.",
    "net_monthly_disposable_gbp": "Remaining monthly disposable income is below our safety buffer.",
    "expense_to_income_ratio": "Regular monthly expenses are high compared to your income.",
    "loan_to_income_ratio": "Requested loan amount is too large relative to your income.",
    "loan_amount_gbp": "Requested loan size carries higher repayment risk.",
    "num_late_payments_24m": "Recent late payments appear on your credit file within the last 24 months.",
    "credit_history": "Past credit repayment history does not meet our current lending criteria.",
    "clean_checking_tier": "Current account balance indicates a low short-term cash buffer.",
    "has_no_checking_account": "No active UK current account was verified for repayments.",
    "savings_balance_gbp": "Savings balance provides a limited cushion for unexpected costs.",
    "months_in_current_job": "Time in current employment is shorter than our stability benchmark.",
    "years_at_current_address": "Time at current address is shorter than our stability benchmark.",
    "previous_application_declined": "A previous recent credit decline indicates elevated borrowing risk.",
    "is_thin_credit_file": "Limited prior borrowing history on your credit file.",
}


def check_application(
    residency_status, num_dependents, region, employment_status,
    months_in_current_job, annual_income_gbp, monthly_expenses_gbp,
    housing_status, years_at_current_address, existing_debt_gbp,
    num_existing_credit_lines, num_late_payments_24m, credit_history,
    has_savings_account, savings_balance_gbp, checking_account_status,
    loan_amount_gbp, loan_term_months, loan_purpose, has_guarantor,
    previous_application_declined, application_channel,
):
    data = pd.DataFrame([{
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

    risk_prob = float(model.predict_proba(data)[0, 1])
    is_declined = risk_prob >= THRESHOLD

    X_encoded = model.named_steps["prep"].transform(data)
    shap_values = explainer.shap_values(X_encoded)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_impacts = list(zip(FEATURE_NAMES, shap_values[0]))
    feature_impacts.sort(key=lambda item: abs(item[1]), reverse=True)

    shap_table = pd.DataFrame([
        {
            "Feature": name.split("__")[-1].replace("_", " ").title(),
            "SHAP Score": round(float(score), 4),
            "Impact": "Increases Risk" if score > 0 else "Lowers Risk",
        }
        for name, score in feature_impacts[:6]
    ])

    if not is_declined:
        status_text = f"APPROVED  |  Estimated Default Risk: {risk_prob:.1%} (Max Allowed: {THRESHOLD:.1%})"
        notice_text = (
            "DECISION: APPROVED\n\n"
            "Your application meets Meridian Credit's affordability and credit checks.\n"
            "No adverse action notice is needed."
        )
    else:
        status_text = f"DECLINED  |  Estimated Default Risk: {risk_prob:.1%} (Cutoff: {THRESHOLD:.1%})"
        reasons = []
        for name, score in feature_impacts:
            if score <= 0:
                continue
            clean_name = name.split("__")[-1]
            if clean_name in REASON_MAP and REASON_MAP[clean_name] not in reasons:
                reasons.append(REASON_MAP[clean_name])
            elif clean_name.startswith("employment_status_"):
                msg = "Current employment type indicates variable income stability."
                if msg not in reasons:
                    reasons.append(msg)
            if len(reasons) == 3:
                break

        if not reasons:
            reasons.append("Overall financial profile did not meet our minimum risk threshold.")

        bullet_points = "\n".join(f"- {r}" for r in reasons)
        notice_text = (
            "DECISION: APPLICATION DECLINED\n\n"
            "Main reasons for this decision:\n"
            f"{bullet_points}\n\n"
            "Your rights:\n"
            "- You can request a human underwriter review within 30 days.\n"
            "- Applying for a smaller loan or reducing existing debt may improve eligibility."
        )

    return status_text, shap_table, notice_text


# Only customize *_dark variables so light mode keeps readable dark text
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="neutral",
).set(
    body_background_fill_dark="#000000",
    block_background_fill_dark="#121212",
    block_border_color_dark="#2a2a2a",
    input_background_fill_dark="#1c1c1c",
    body_text_color_dark="#f5f5f5",
)

with gr.Blocks(
    title="Meridian Credit Risk App",
    theme=theme,
    js="() => { document.body.classList.add('dark'); }",
) as demo:
    gr.Markdown("# Meridian Credit — Loan Risk & Explainability Portal")
    gr.Markdown("Fill in the 3 simple steps below (or click one of the **Quick Demo Scenarios** at the bottom) and press **Assess Application**.")

    with gr.Tabs():
        with gr.Tab("Step 1: Personal & Job"):
            with gr.Row():
                residency = gr.Dropdown(["UK Citizen", "Settled Status", "Visa Holder"], value="UK Citizen", label="Residency Status", info="Current UK residency status")
                region = gr.Dropdown(["London", "South East", "Midlands", "North West", "Yorkshire", "Scotland", "South West", "North East", "Wales", "Northern Ireland"], value="Midlands", label="UK Region", info="Where the applicant lives")
                housing = gr.Dropdown(["Renting", "Mortgage", "Living with family", "Owner"], value="Mortgage", label="Housing Status", info="Current living arrangement")
            with gr.Row():
                addr_years = gr.Slider(0, 30, value=4.0, step=0.5, label="Years at Current Address")
                dependents = gr.Slider(0, 8, value=1, step=1, label="Number of Dependents")
            with gr.Row():
                emp_status = gr.Dropdown(["Permanent", "Self-employed", "Fixed-term", "Zero-hours", "Unemployed", "Retired"], value="Permanent", label="Employment Type", info="Select Unemployed/Retired if not working")
                job_months = gr.Slider(0, 240, value=36, step=1, label="Months in Current Job", info="Set to 0 if unemployed or retired")

        with gr.Tab("Step 2: Income & Credit"):
            with gr.Row():
                income = gr.Slider(5000, 150000, value=38000, step=500, label="Annual Income (£)", info="Gross yearly income before tax")
                expenses = gr.Slider(200, 8000, value=1200, step=50, label="Monthly Expenses (£)", info="Rent, bills, food, and regular outgoings")
                existing_debt = gr.Slider(0, 60000, value=3500, step=250, label="Existing Debt (£)", info="Total unpaid loans and credit cards")
            with gr.Row():
                cred_hist = gr.Dropdown(["All paid on time", "Existing credits paid", "No credits taken", "Past delays", "Critical account"], value="All paid on time", label="Credit History")
                credit_lines = gr.Slider(0, 15, value=3, step=1, label="Existing Credit Lines")
                late_payments = gr.Slider(0, 10, value=0, step=1, label="Late Payments (Last 24 Months)")
            with gr.Row():
                chk_stat = gr.Dropdown(["500 GBP or more", "0 to 500 GBP", "Overdrawn", "No account"], value="500 GBP or more", label="Checking Account Balance")
                has_sav = gr.Radio(["Yes", "No"], value="Yes", label="Has Savings Account?")
                sav_bal = gr.Slider(0, 50000, value=4500, step=250, label="Savings Balance (£)")

        with gr.Tab("Step 3: Loan Request"):
            with gr.Row():
                loan_amt = gr.Slider(1000, 40000, value=8000, step=500, label="Requested Loan Amount (£)")
                loan_term = gr.Slider(6, 60, value=36, step=6, label="Loan Term (Months)")
                purpose = gr.Dropdown(["Car", "Debt consolidation", "Home improvement", "Education", "Other", "Business", "Holiday", "Medical"], value="Home improvement", label="Loan Purpose")
            with gr.Row():
                guarantor = gr.Radio(["Yes", "No"], value="No", label="Has Guarantor?")
                prev_dec = gr.Radio(["Yes", "No"], value="No", label="Previously Declined?")
                channel = gr.Dropdown(["Online", "Broker", "Branch", "Phone"], value="Online", label="Application Channel")

    with gr.Row():
        submit_btn = gr.Button("Assess Application", variant="primary")
        reset_btn = gr.ClearButton(value="Clear / Reset")

    result_box = gr.Textbox(label="Lending Decision", interactive=False)

    with gr.Row():
        shap_out = gr.Dataframe(label="Underwriter View: Top SHAP Factors")
        notice_out = gr.Textbox(label="Applicant View: Plain-English Notice", lines=10, interactive=False)

    inputs = [
        residency, dependents, region, emp_status, job_months, income,
        expenses, housing, addr_years, existing_debt, credit_lines,
        late_payments, cred_hist, has_sav, sav_bal, chk_stat, loan_amt,
        loan_term, purpose, guarantor, prev_dec, channel,
    ]
    reset_btn.add(inputs + [result_box, shap_out, notice_out])
    submit_btn.click(fn=check_application, inputs=inputs, outputs=[result_box, shap_out, notice_out])

    gr.Examples(
        label="Quick Demo Scenarios (Click a row to load: 1. Clean Approval | 2. High-Risk Decline | 3. Business Default Edge Case)",
        inputs=inputs,
        examples=[
            ["UK Citizen", 1, "Midlands", "Permanent", 48, 42000, 1100, "Mortgage", 5.0, 2000, 3, 0, "All paid on time", "Yes", 6000, "500 GBP or more", 7500, 36, "Car", "No", "No", "Online"],
            ["Visa Holder", 2, "London", "Zero-hours", 4, 21000, 1450, "Renting", 0.5, 14000, 5, 3, "Past delays", "No", 0, "Overdrawn", 12000, 24, "Debt consolidation", "No", "Yes", "Broker"],
            ["UK Citizen", 0, "London", "Permanent", 34, 116480, 4570, "Mortgage", 8.5, 0, 3, 0, "Existing credits paid", "No", 0, "0 to 500 GBP", 19800, 48, "Business", "No", "No", "Online"],
        ],
    )

if __name__ == "__main__":
    demo.launch()