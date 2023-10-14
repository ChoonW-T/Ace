import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading dataset to a pandas Dataframe
credit_card_data = pd.read_csv('creditcard.csv')

#data will have to be taken from the following website:  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

#check for missing values
credit_card_data.isnull().sum()

#distribution for legit and fraudulent transactions
credit_card_data['Class'].value_counts()

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

#statistical measures of the data
legit.Amount.describe()
fraud.Amount.describe()

#compare the values for both transactions
credit_card_data.groupby('Class').mean()

#bootstrapping 492 values
legit_sample = legit.sample(n=492)

#making a new dataset with bootstrapped legit values
new_data = pd.concat([legit_sample, fraud], axis=0)

#data is now normally distributed
new_data['Class'].value_counts()

new_data.groupby('Class').mean()

X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']

#split data to training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#creating logistic regression model
model = LogisticRegression()

model.fit(X_train, Y_train)

#evaluating model with training data
X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, Y_train)
print("Training Data Accuracy:", training_data_accuracy)

#evaluating model with testing data
X_test_predict = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_predict, Y_test)

print("Testing Data Accuracy:", testing_data_accuracy)