import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import imblearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import sklearn.externals
import joblib
import streamlit as st

ecommerce = pd.read_csv('cleaned_data.csv')

df1 = ecommerce.copy()

target_mapper = {'satisfied' : 0, 'not satisfied' : 1}
order_mapper = {'delivered' : 0 , 'canceled' : 1}
payment_mapper = {'credit_cart' : 1, 'coupon' : 2, 'voucher' :3, 'debit cart':4}
price_mapper = {'<50' : 1, '50 - 100' : 2, '100 - 250' : 3, '250 - 500' : 4 , '500 - 1000' : 5, '1000 - 2000' :6, '2000 - 3000' : 7, '3000 - 4000' :8, '4000 - 5000' : 9, '>5000' : 10}
photos_mapper = {'<5' : 1, '5 - 10' : 2, '10 -15' : 3, '>15' : 4}
time_mapper = {'1' : 1, '1 - 5' : 2, '5 - 10' : 3, '10 - 20' : 4, '20 - 30' : 5, '30 - 40' : 6, '40 - 50' : 7, '>50' : 8}
delivery_on_time_mapper = {'No' : 1, 'Yes' : 2}
delivered_hour_mapper = {'Morning' : 1, 'Midday' : 2, 'Evening' : 3, 'Night' : 4}
delivered_season_mapper = {'Summer' : 0, 'Winter' : 1}
seller_count_mapper = {'<10' : 1, '10 - 50' : 2, '50 - 250' : 3, '250 - 500' : 4, '500 - 1000' : 5, '1000 - 2000' : 6, '2000 - 3000' : 7, '3000 - 4000' : 8, '4000 - 5000' : 9, '>5000' : 10}

df1['order_status'] = df1.order_status.map(order_mapper)
df1['review_score'] = df1.review_score.map(target_mapper)
df1['payment_type'] = df1.payment_type.map(payment_mapper)
df1['price'] = df1.price.map(price_mapper)
df1['photos_count'] = df1.photos_count.map(photos_mapper)
df1['time_different_hour'] = df1.time_different_hour.map(time_mapper)
df1['delivered_on_time'] = df1.delivered_on_time.map(delivery_on_time_mapper)
df1['delivered_hour'] = df1.delivered_hour.map(delivered_hour_mapper)
df1['delivered_season'] = df1.delivered_season.map(delivered_season_mapper)
df1['seller_count'] = df1.seller_count.map(seller_count_mapper)
df1['photos_count'] = df1['photos_count'].astype(int)


encode = ['order_status', 'payment_type', 'price', 'photos_count','time_different_hour', 'delivered_on_time', 'delivered_hour', 'delivered_season', 'seller_count']

for col in encode:
    dummy = pd.get_dummies(df1[col], prefix=col)
    df1 = pd.concat([df1,dummy], axis=1)
    del df1[col]

X = df1.drop('review_score', axis=1)
y = df1['review_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_smoted, y_smoted = SMOTE(random_state=42).fit_resample(X,y)
classifier = RandomForestClassifier().fit(X_smoted, y_smoted)
clf = RandomForestClassifier()
clf.fit(X_smoted, y_smoted)

st.write("""
# e - commerce customer prediction

This app predicts the ** Customer Satisfaction ** 
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://...)
""")


def user_input_features():
    order_status = st.sidebar.selectbox('Orders', ('delivered', 'canceled'))
    payment_type = st.sidebar.selectbox('Payment Type', ('credit_cart', 'coupon', 'voucher', 'debit cart'))
    price = st.sidebar.selectbox('Price', (
    '<50', '50 - 100', '100 - 250', '250 - 500', '500 - 1000', '1000 - 2000', '2000 - 3000', '3000 - 4000',
    '4000 - 5000', '>5000'))
    photos_count = st.sidebar.selectbox('Photos Count', ('<5', '5 - 10', '10 -15', '>15'))
    time_different_hour = st.sidebar.selectbox('Time Different Hour', (
    '<1', '1 - 5', '5 - 10', '10 - 20', '20 - 30', '30 - 40', '40 - 50', '>50'))
    delivered_on_time = st.sidebar.selectbox('Delivery on Time', ('No', 'Yes'))
    delivered_hour = st.sidebar.selectbox('Delivered Hour', ('Morning', 'Midday', 'Evening', 'Night'))
    delivered_season = st.sidebar.selectbox('Delivered Hour', ('Winter', 'Summer'))
    seller_count = st.sidebar.selectbox('Seller Count', (
    '<10', '10 - 50', '50 - 250', '250 - 500', '500 - 1000', '1000 - 2000', '2000 - 3000', '3000 - 4000', '4000 - 5000',
    '>5000'))
    data = {'order_status': [order_status],
            'payment_type': [payment_type],
            'price': [price],
            'photos_count': [photos_count],
            'time_different_hour': [time_different_hour],
            'delivered_on_time': [delivered_on_time],
            'delivered_hour': [delivered_hour],
            'delivered_season': [delivered_season],
            'seller_count': [seller_count]}
    features = pd.DataFrame(data)
    return features
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
st.write('Awaiting CSV file to be uploaded. Currently using example input parametrs (show below)')
st.write(df)
# Apply model to make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
st.subheader('Prediction')

customer_satisfy = np.array(['not satisfied', 'satisfied'])
st.write(customer_satisfy[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)