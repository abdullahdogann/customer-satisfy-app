import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import imblearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import bz2
import pickle
import _pickle as cPickle

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

#pickle.dump(clf, open('ecomerce_clf_pkl', 'wb'))

def compressed_pickle(title,data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data,f)
compressed_pickle('ecomerce_clf_pkl', clf)
print(X.shape)