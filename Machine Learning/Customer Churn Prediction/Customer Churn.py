import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import numpy as np

df = pd.read_csv("customer_churn.csv")
df.drop('customerID', axis='columns', inplace=True)

df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()]

df.iloc[488]
df1=df[df.TotalCharges!='']
df1.TotalCharges= pd.to_numeric(df.TotalCharges)

tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.hist([tenure_churn_yes, tenure_churn_no], color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend

mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges

plt.hist([mc_churn_yes, mc_churn_no], color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend

df1.replace['No internet service', 'No', inplace=True]
df1.replace['No phone service', 'No', inplace=True]

yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes':1,'No':0}, inplace=True)

df1['gender'].replace({'Female':1,'Male':0}, inplace=True)

df2= pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

scalar = MinMaxScaler()
df2[cols_to_scale] = scalar.fit_transform(df2[cols_to_scale])

X = df2.drop('Churn', axis='columns')
Y = df2['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

model = keras.Sequential([
    keras.layer.Dense(20, input_shape=(26,), activation='relu'),
    keras.layer.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metric=['accuracy'])

model.fit(X_train,Y_train, epochs=50)

model.evaluate(X_test,Y_test)

YP = model.predict(X_test)

Y_pred = []
for element in YP:
    if element > 0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)

print(classification_report(Y_test, Y_pred))

cm = tf.math.confusion_matrix(labels=Y_test, predictions=Y_pred)

sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')