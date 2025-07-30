import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling and Background ---
def set_styles():
    """
    Applies custom CSS for styling. This final version specifically targets
    the slider value labels for visibility.
    """
    st.markdown(
        """
        <style>
        /* Target the root app container for the background */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1609429019995-8c40f49535a5?q=80&w=3269&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }

        /* Make the main content area transparent */
        [data-testid="stAppViewContainer"] > .main {
            background-color: transparent;
        }

        /* Style for the content columns to create a "card" effect */
        [data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] {
            background-color: rgba(0, 0, 0, 0.75);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Style headers and text to be white with a shadow for pop */
        h1, h2, h3, p, label {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
        }

        /* Style for the prediction result container */
        .prediction-card {
            background-color: rgba(0, 0, 0, 0.85);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Style for the metric text to ensure it's white */
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: white !important;
            text-shadow: none;
        }

        /* Style for slider track and labels */
        [data-testid="stSlider"] .st-emotion-cache-134s3z3 {
            background: rgba(255, 255, 255, 0.2) !important;
        }
        /* Force slider value labels (e.g., -1, 8) to be visible */
        [data-testid="stTickBar"] > div {
            color: yellow !important;
            text-shadow: 1px 1px 2px black; /* Add shadow to numbers too */
        }
        /* This targets the slider value "bubble" */
        [data-testid="stSlider"] span[role="slider"] {
            color: yellow !important;
            text-shadow: 1px 1px 2px black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the styles
set_styles()


@st.cache_resource
def load_model():
    """Loads the saved model file, caching it for performance."""
    try:
        model = joblib.load('credit_default_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'credit_default_model.pkl' not found. Please ensure it's in the correct directory.")
        return None

model = load_model()

# --- App Header ---
if model:
    st.title("Credit Card Default Prediction")
    st.write("This app predicts whether a credit card holder is likely to **default**, meaning they will fail to make a required payment next month.")
    st.write("Enter customer details below to predict the likelihood of payment default.")

    # --- Input Section ---
    st.header("Customer Data Input")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("👤 Customer Profile")
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, value=50000, step=1000)
        sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}.get(x))
        marriage = st.selectbox("Marriage", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}.get(x))
        age = st.slider("Age", 21, 79, 35)

    with col2:
        st.subheader("📜 Payment History (Last 6 Months)")
        st.caption("(-2=no consumption, -1=paid duly, 0=revolving, 1-9=delayed)")
        pay_0 = st.slider("Status in Sep, 2005 (PAY_0)", -2, 9, 0)
        pay_2 = st.slider("Status in Aug, 2005 (PAY_2)", -2, 9, 0)
        pay_3 = st.slider("Status in Jul, 2005 (PAY_3)", -2, 9, 0)
        pay_4 = st.slider("Status in Jun, 2005 (PAY_4)", -2, 9, 0)
        pay_5 = st.slider("Status in May, 2005 (PAY_5)", -2, 9, 0)
        pay_6 = st.slider("Status in Apr, 2005 (PAY_6)", -2, 9, 0)

    with col3:
        st.subheader("💵 Account Balances (Last 6 Months)")
        bill_amt1 = st.number_input("Bill in Sep (BILL_AMT1)", min_value=0, value=20000)
        pay_amt1 = st.number_input("Paid in Sep (PAY_AMT1)", min_value=0, value=1000)
        st.markdown("---")
        bill_amt2 = st.number_input("Bill in Aug (BILL_AMT2)", min_value=0, value=18000)
        pay_amt2 = st.number_input("Paid in Aug (PAY_AMT2)", min_value=0, value=1000)
        st.markdown("---")
        bill_amt3 = st.number_input("Bill in Jul (BILL_AMT3)", min_value=0, value=19000)
        pay_amt3 = st.number_input("Paid in Jul (PAY_AMT3)", min_value=0, value=1000)
        st.markdown("---")
        bill_amt4 = st.number_input("Bill in Jun (BILL_AMT4)", min_value=0, value=17000)
        pay_amt4 = st.number_input("Paid in Jun (PAY_AMT4)", min_value=0, value=1000)
        st.markdown("---")
        bill_amt5 = st.number_input("Bill in May (BILL_AMT5)", min_value=0, value=15000)
        pay_amt5 = st.number_input("Paid in May (PAY_AMT5)", min_value=0, value=1000)
        st.markdown("---")
        bill_amt6 = st.number_input("Bill in Apr (BILL_AMT6)", min_value=0, value=15000)
        pay_amt6 = st.number_input("Paid in Apr (PAY_AMT6)", min_value=0, value=1000)

    # --- Prediction Logic ---
    def prepare_input_data(data):
        input_df = pd.DataFrame([data])
        bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
        pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
        pay_status_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
        input_df['TOTAL_BILL_AMT'] = input_df[bill_cols].sum(axis=1)
        input_df['TOTAL_PAY_AMT'] = input_df[pay_amt_cols].sum(axis=1)
        input_df['PAYMENT_RATIO'] = input_df['TOTAL_PAY_AMT'] / (input_df['TOTAL_BILL_AMT'] + 1)
        input_df['AVG_DELAY'] = input_df[pay_status_cols].mean(axis=1)
        input_df['SEX_2'] = 1 if data['SEX'] == 2 else 0
        for i in [2, 3, 4]: input_df[f'EDUCATION_{i}'] = 1 if data['EDUCATION'] == i else 0
        for i in [2, 3]: input_df[f'MARRIAGE_{i}'] = 1 if data['MARRIAGE'] == i else 0
        return input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    st.markdown("---")
    
    if st.button("Analyze Customer Risk", use_container_width=True, type="primary"):
        user_data = {
            'LIMIT_BAL': limit_bal, 'AGE': age, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage,
            'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3, 'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
            'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3, 'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6,
        }
        processed_input = prepare_input_data(user_data)
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]

        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.header("Prediction Result")
        if prediction == 1:
            st.error("🔴 High Risk: Customer is likely to Default.", icon="🚨")
            st.metric(label="Default (Not paying) Probability", value=f"{prediction_proba[1]*100:.2f}%")
            st.caption("This is the model's confidence score that the customer will default.")

        else:
            st.success("🟢 Low Risk: Customer is likely to Pay.", icon="✅")
            st.metric(label="Default (Not paying) Probability", value=f"{prediction_proba[1]*100:.2f}%")
            st.caption("This is the model's confidence score that the customer will default.")

        st.markdown('</div>', unsafe_allow_html=True)
