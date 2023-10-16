import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset into a pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')

# Link to dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Check for any missing values in the dataset
missing_values = credit_card_data.isnull().sum()
if missing_values.any():
    print("Missing values detected:", missing_values)

# Display distribution of legitimate and fraudulent transactions
print(credit_card_data['Class'].value_counts())

# Filter out legit and fraud transactions for separate analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Display statistical measures of the transaction amounts for both classes
print("Legit Transactions Summary:\n", legit.Amount.describe())
print("\nFraud Transactions Summary:\n", fraud.Amount.describe())

# Compare mean values across all columns for both classes
print("\nMean Values by Class:\n", credit_card_data.groupby('Class').mean())

# Bootstrap (sample) 492 values from the legitimate transactions for balanced dataset creation
legit_sample = legit.sample(n=492)

# Combine bootstrapped legit values with original fraud values to get a balanced dataset
new_data = pd.concat([legit_sample, fraud], axis=0)

# Confirm that the data is now balanced
print("\nBalanced Class Distribution:\n", new_data['Class'].value_counts())
print("\nMean Values in Balanced Dataset by Class:\n", new_data.groupby('Class').mean())

# Separate features (X) from target variable (Y)
X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']

# Split the balanced dataset into training and testing sets, ensuring both have a similar ratio of legit and fraud instances
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize and train a logistic regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter for better convergence
model.fit(X_train, Y_train)

# Evaluate model's accuracy on the training data
X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, Y_train)
print("\nTraining Data Accuracy:", training_data_accuracy)

# Evaluate model's accuracy on the testing data
X_test_predict = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_predict, Y_test)
print("Testing Data Accuracy:", testing_data_accuracy)