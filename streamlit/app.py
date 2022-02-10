import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import sklearn.externals
import joblib

st.write("""
# e - commerce customer prediction

This app predicts the ** Customer Satisfaction ** 
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/abdullahdogann/customer-satisfy-app/blob/main/streamlit/ecommerce_data.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file")
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        order_status = st.sidebar.selectbox('Orders', ('delivered','canceled'))
        payment_type = st.sidebar.selectbox('Payment Type', ('credit_cart', 'coupon', 'voucher','debit cart'))
        price = st.sidebar.selectbox('Price', ('<50', '50 - 100','100 - 250','250 - 500','500 - 1000','1000 - 2000','2000 - 3000','3000 - 4000','4000 - 5000','>5000'))
        photos_count = st.sidebar.selectbox('Photos Count', ('<5','5 - 10', '10 -15','>15'))
        time_different_hour = st.sidebar.selectbox('Time Different Hour', ('<1','1 - 5','5 - 10','10 - 20','20 - 30','30 - 40','40 - 50','>50'))
        delivered_on_time = st.sidebar.selectbox('Delivery on Time', ('No', 'Yes'))
        delivered_hour = st.sidebar.selectbox('Delivered Hour', ('Morning','Midday','Evening','Night'))
        delivered_season = st.sidebar.selectbox('Delivered Hour', ('Winter', 'Summer'))
        seller_count = st.sidebar.selectbox('Seller Count', ('<10','10 - 50','50 - 250','250 - 500','500 - 1000','1000 - 2000','2000 - 3000','3000 - 4000','4000 - 5000','>5000'))
        data = {'order_status' : [order_status],
                 'payment_type' : [payment_type],
                 'price' : [price],
                 'photos_count' : [photos_count],
                 'time_different_hour' : [time_different_hour],
                 'delivered_on_time' : [delivered_on_time],
                 'delivered_hour' : [delivered_hour],
                 'delivered_season' : [delivered_season],
                 'seller_count' : [seller_count]}
        features = pd.DataFrame(data)
        return  features
    input_df = user_input_features()
ecommerce_raw = pd.read_csv('cleaned_data.csv')
ecommerce = ecommerce_raw.drop(columns=['review_score'])
df = pd.concat([input_df, ecommerce], axis=0)

encode = ['order_status', 'payment_type', 'price', 'photos_count','time_different_hour', 'delivered_on_time', 'delivered_hour', 'delivered_season', 'seller_count']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]
df.drop('Unnamed: 0', axis=1, inplace = True)

# Displays the user input features
st.subheader('User Input features')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parametrs (show below)')
    st.write(df)

# Reads in saved classification model
load_clf = joblib.load('ecomerce_clf_pkl.pbz2')

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')

customer_satisfy = np.array(['not satisfied', 'satisfied'])
st.write(customer_satisfy[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

