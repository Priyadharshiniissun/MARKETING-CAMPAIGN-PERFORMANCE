import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np


revenue_models = joblib.load("Regr_models.pkl")

revenue_model = revenue_models["random_forest"]

profit_model = joblib.load("Class_model.pkl")

st.title("📊 Campaign Revenue & Profit Prediction")

st.write("Enter campaign details to predict revenue and profitability")

duration = st.number_input("Campaign Duration")

impressions = st.number_input("Impressions")

clicks = st.number_input("Clicks")

leads = st.number_input("Leads")

conversions = st.number_input("Conversions")

spend = st.number_input("Acquisition Cost")

engagement = st.number_input("Engagement Score")

roi = st.number_input("ROI")

channel = st.multiselect("Marketing Channel",["Email","Facebook","Google","Instagram","YouTube","WhatsApp"])

campaign_type = st.selectbox("Campaign Type",["Influencer","Paid Ads","SEO","Email","Social Media"])

audience = st.selectbox("Target Audience",["Premium Shoppers","Youth","Working Women","Tier 2 City Customers","College Students"])

language = st.selectbox("Language",["Hindi","English","Tamil","Bengali"])

customer_segment = st.selectbox("Customer Segment",["Premium Shoppers","Youth","Working Women","Tier 2 City Customers","College Students"])

input_data = pd.DataFrame({"Duration":[duration],"Impressions":[impressions],
                            "Clicks":[clicks],"Leads":[leads],"Conversions":[conversions],
                            "Acquisition_Cost":[spend],"Engagement_Score":[engagement],"ROI":[roi],
                            "Email": [1 if "Email" in channel else 0],
                            "Facebook": [1 if "Facebook" in channel else 0],
                            "Google": [1 if "Google" in channel else 0],
                            "Instagram": [1 if "Instagram" in channel else 0],
                            "YouTube": [1 if "YouTube" in channel else 0],
                            "WhatsApp": [1 if "WhatsApp" in channel else 0],
                            "Campaign_Type": [campaign_type],
                            "Target_Audience": [audience],
                            "Language": [language],"Customer_Segment":[customer_segment]})


if st.button("Predict"):
    st.subheader("Input Summary")
    st.dataframe(input_data)
    revenue_log = revenue_model.predict(input_data)
     
    revenue = np.expm1(revenue_log)
     
    profit = profit_model.predict(input_data)[0]
     
    st.subheader("Prediction Result")

    col1,col2 = st.columns(2)

    with col1:st.metric("Predicted Revenue",f"₹ {revenue[0]:,.2f}")


    with col2:
        if profit==1:
            st.success("Prediction: PROFIT")
        else:
            st.error("Prediction: LOSS")

    result_df = pd.DataFrame({"Metric":["Revenue","Spend"],"Value":[revenue[0],spend]})

    fig = px.bar(result_df,x="Metric",y="Value",title="Campaign Performance")

    st.plotly_chart(fig)
     
