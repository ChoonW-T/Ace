from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

df = pd.read_excel("processed_data.xlsx", engine='openpyxl')

# Data Sampling
# Splitting dataset into training and test samples to validate model's performance on unseen data
train_df = df.copy()

# Separate target variable from independent variables
y = df['churn']
X = df.drop(columns=['id', 'churn'])
print(X.shape)
print(y.shape)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Model Training
# Using Random Forest classifier due to its ensemble nature, making predictions based on multiple decision trees
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
rf_predictions = model.predict(X_test)
rf_accuracy = metrics.accuracy_score(y_test, rf_predictions)
print(f"Random Forest Regression Accuracy: {rf_accuracy}")

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = metrics.accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

# 2. XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Avoid deprecation warning
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = metrics.accuracy_score(y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy}")

# 3. Support Vector Machines
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = metrics.accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy}")

# Evaluation
# Generating confusion matrix values for calculating performance metrics
tn, fp, fn, tp = metrics.confusion_matrix(y_test, rf_predictions).ravel()

# Displaying counts of churn values in test dataset
print(y_test.value_counts())

# Displaying performance metrics to evaluate model's accuracy, precision, and recall
print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}\n")
print(f"Accuracy: {metrics.accuracy_score(y_test, rf_predictions)}")
print(f"Precision: {metrics.precision_score(y_test, rf_predictions)}")
print(f"Recall: {metrics.recall_score(y_test, rf_predictions)}")

# Identifying feature importances to understand influential drivers for predictions
feature_importances = pd.DataFrame({
    'features': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=True).reset_index()

# Visualizing feature importances to help in model interpretation
plt.figure(figsize=(15, 25))
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances['importance'], color='b', align='center')
plt.yticks(range(len(feature_importances)), feature_importances['features'])
plt.xlabel('Importance')
plt.show()

# Predicting probabilities to understand the likelihood of a churn
proba_predictions = model.predict_proba(X_test)
probabilities = proba_predictions[:, 1]

# Preparing test dataset to save predictions and their probabilities
X_test = X_test.reset_index()
X_test.drop(columns='index', inplace=True)
X_test['churn'] = rf_predictions.tolist()
X_test['churn_probability'] = probabilities.tolist()

# Saving predictions with probabilities for further analysis
X_test.to_csv('out_of_sample_data_with_predictions.csv')