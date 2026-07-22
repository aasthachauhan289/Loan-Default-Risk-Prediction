# Loan Default Risk Prediction using Streamlit

# Import required Libraries
import streamlit as st
import pandas as pd
import joblib

# Configure the Streamlit page
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load the trained Random Forest model
model = joblib.load("models/loan_default_rf_model.pkl")

# Load the StandardScaler used during model training
scaler = joblib.load("models/loan_default_scaler.pkl")

# Application title
st.title("🏦 Loan Default Risk Prediction")

# Project description
st.write("""
This application predicts whether a customer is likely to default on a loan
using a trained Random Forest Machine Learning model.
Please enter the customer details below and click the Predict button.
""")

st.markdown("---")

# Customer Information 

st.header("Customer Information")

#  Personal Details 

st.subheader("Personal Details")

# Customer Age
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

# Customer Education Level
education = st.selectbox(
    "Education",
    [
        "Bachelor",
        "High School",
        "Master's",
        "PhD"
    ]
)

# Customer Employment Type
employment_type = st.selectbox(
    "Employment Type",
    [
        "Full-time",
        "Part-time",
        "Self-employed",
        "Unemployed"
    ]
)

# Customer Marital Status
marital_status = st.selectbox(
    "Marital Status",
    [
        "Divorced",
        "Married",
        "Single"
    ]
)

# Whether the customer has dependents
has_dependents = st.radio(
    "Has Dependents?",
    [
        "Yes",
        "No"
    ]
)

st.markdown("---")

#  Financial Details 

st.subheader("Financial Details")

# Customer Annual Income
income = st.number_input(
    "Annual Income ($)",
    min_value=0,
    value=50000,
    step=1000
)

# Customer Credit Score
credit_score = st.slider(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

# Number of Months Employed
months_employed = st.number_input(
    "Months Employed",
    min_value=0,
    value=60
)

# Total Number of Credit Lines
num_credit_lines = st.number_input(
    "Number of Credit Lines",
    min_value=0,
    value=2
)

# Loan Interest Rate
interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=40.0,
    value=10.0
)

# Loan Term in Months
loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=36
)

# Debt-to-Income Ratio
dti_ratio = st.slider(
    "Debt-to-Income Ratio",
    min_value=0.00,
    max_value=1.00,
    value=0.50
)

st.markdown("---")


#  Loan Details 

st.subheader("Loan Details")

# Requested Loan Amount
loan_amount = st.number_input(
    "Loan Amount ($)",
    min_value=1000,
    value=100000,
    step=1000
)

# Purpose of the Loan
loan_purpose = st.selectbox(
    "Loan Purpose",
    [
        "Auto",
        "Business",
        "Education",
        "Home",
        "Other"
    ]
)

# Whether the customer already has a mortgage
has_mortgage = st.radio(
    "Has Mortgage?",
    [
        "Yes",
        "No"
    ]
)

# Whether the customer has a co-signer
has_cosigner = st.radio(
    "Has Co-Signer?",
    [
        "Yes",
        "No"
    ]
)

st.markdown("---")


# Convert Yes/No values into numerical values
has_mortgage = 1 if has_mortgage == "Yes" else 0
has_dependents = 1 if has_dependents == "Yes" else 0
has_cosigner = 1 if has_cosigner == "Yes" else 0

# Create Predict Button
if st.button("Predict Loan Status"):

    # Create input data in DataFrame format
    input_data = pd.DataFrame({

        # Numerical Features
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],

        # Education Encoding
        "Education_High School": [1 if education == "High School" else 0],
        "Education_Master's": [1 if education == "Master's" else 0],
        "Education_PhD": [1 if education == "PhD" else 0],

        # Employment Type Encoding
        "EmploymentType_Part-time": [1 if employment_type == "Part-time" else 0],
        "EmploymentType_Self-employed": [1 if employment_type == "Self-employed" else 0],
        "EmploymentType_Unemployed": [1 if employment_type == "Unemployed" else 0],

        # Marital Status Encoding
        "MaritalStatus_Married": [1 if marital_status == "Married" else 0],
        "MaritalStatus_Single": [1 if marital_status == "Single" else 0],

        # Binary Features
        "HasMortgage_Yes": [has_mortgage],
        "HasDependents_Yes": [has_dependents],

        # Loan Purpose Encoding
        "LoanPurpose_Business": [1 if loan_purpose == "Business" else 0],
        "LoanPurpose_Education": [1 if loan_purpose == "Education" else 0],
        "LoanPurpose_Home": [1 if loan_purpose == "Home" else 0],
        "LoanPurpose_Other": [1 if loan_purpose == "Other" else 0],

        # Co-Signer Feature
        "HasCoSigner_Yes": [has_cosigner]

    })

    # Scale the input data using the saved StandardScaler
    input_scaled = scaler.transform(input_data)

    # Convert back to DataFrame to keep feature names
    input_scaled = pd.DataFrame(
        input_scaled,
        columns=input_data.columns
    )

    # Predict loan status
    prediction = model.predict(input_scaled)

    # Predict probability
    probability = model.predict_proba(input_scaled)
  
    # Probability of loan default
    default_probability = probability[0][1]

    st.markdown("---")

    # Display Prediction Result
    st.header("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Loan Default")

    else:
        st.success("✅ Non Default")

    # Display Default Probability
    st.write(f"**Default Probability:** {default_probability:.2%}")

    # Display Risk Category
    if default_probability < 0.30:
        st.success("🟢 Risk Level : Low Risk")

    elif default_probability < 0.70:
        st.warning("🟡 Risk Level : Medium Risk")

    else:
        st.error("🔴 Risk Level : High Risk")

    # Display Prediction Summary
    st.markdown("---")

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Income",
            "Loan Amount",
            "Credit Score",
            "Interest Rate",
            "Loan Term",
            "DTI Ratio"
        ],
        "Value": [
            age,
            income,
            loan_amount,
            credit_score,
            interest_rate,
            loan_term,
            dti_ratio
        ]
    })

    st.dataframe(summary, width="stretch")