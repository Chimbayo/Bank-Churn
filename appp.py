import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and data
model = joblib.load("lgb_model.pkl")
df = pd.read_csv("malawi_bank_customers.csv.csv")  # Your customer dataset

# Define input features expected by the model
model_input_columns = [
    'Age', 'Gender', 'District', 'Region', 'Location_Type', 'Customer_Type',
    'Employment_Status', 'Income_Level', 'Education_Level', 'Tenure',
    'Balance', 'Credit_Score', 'Outstanding_Loans', 'Num_Of_Products',
    'Mobile_Banking_Usage', 'Number_of _Transactions_per/Month',
    'Num_Of_Complaints', 'Proximity_to_NearestBranch_or_ATM (km)',
    'Mobile_Network_Quality', 'Owns_Mobile_Phone'
]

# Predict churn and probabilities
X = df[model_input_columns]
proba = model.predict_proba(X)
df['Churn_Probability'] = proba[:, 1]
df['Prediction'] = (df['Churn_Probability'] >= 0.5).astype(int)

# Sidebar filters
st.sidebar.title("Filters")
min_prob = st.sidebar.slider("Min Churn Probability", 0.0, 1.0, 0.7)
region_filter = st.sidebar.selectbox("Select Region", ["All"] + sorted(df['Region'].unique().tolist()))

# Filtered dataframe
filtered_df = df[df['Churn_Probability'] >= min_prob]
if region_filter != "All":
    filtered_df = filtered_df[filtered_df['Region'] == region_filter]

# KPI Metrics
st.title("📈 Bank Customer Churn Dashboard")
st.subheader("Overview")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Customers", len(df))
kpi2.metric("Churned", df['Prediction'].sum())
kpi3.metric("Not Churned", len(df) - df['Prediction'].sum())

# Pie chart of churn
st.subheader("Churn Distribution")
pie_chart = px.pie(df, names='Prediction', title='Churn vs Not Churn', 
                   labels={0: 'Stayed', 1: 'Left'},
                   color='Prediction', color_discrete_map={0: 'green', 1: 'red'})
st.plotly_chart(pie_chart)

# Probability distribution
st.subheader("Churn Probability Histogram")
hist_fig = px.histogram(df, x='Churn_Probability', nbins=30, title="Churn Probability Distribution")
st.plotly_chart(hist_fig)

# Filtered Customer Table
st.subheader("High Risk Customers")
st.dataframe(filtered_df)

# Download filtered high-risk customers
st.download_button(
    label="🔧 Download High-Risk Customers",
    data=filtered_df.to_csv(index=False),
    file_name="high_risk_customers.csv",
    mime="text/csv"
)

# Optional save to file
filtered_df.to_csv("high_risk_customers.csv", index=False)

st.success("Dashboard loaded successfully.")