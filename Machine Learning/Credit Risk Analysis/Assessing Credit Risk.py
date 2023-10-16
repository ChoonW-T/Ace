import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import warnings

# Supress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the dataset and display its shape
df = pd.read_csv('Data.csv', index_col=0)
print(df.shape)
df.head(10)

# Fill missing values in 'loan_status' with the most frequent value
df['loan_status'].fillna(df['loan_status'].mode()[0], inplace=True)

# Backup original dataframe
df_backup = df.copy()

# Filter columns with more than 10 missing values and drop rows with any missing values
df = df.loc[:, df.isnull().sum() <= 10]
df.dropna(inplace=True)

# Create a binary representation for 'loan_status'
df["loan_dummy"] = np.where(df["loan_status"] == "Charged Off", 1, 0)

# Visualize loan status distribution
df["loan_dummy"].value_counts().plot(kind='bar')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.title('Distribution of loan status')
plt.show()

# Visualize loan amount distribution by grade
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sns.barplot(data=df, y='loan_amnt', x='grade', order=grade_order)
plt.show()

# Define predictor variables and target variable
X = df[['loan_amnt', 'int_rate', 'annual_inc', 'total_pymnt', 'installment', 'total_rec_int', 'last_pymnt_amnt']]
y = df['loan_dummy'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Logistic Regression model and make predictions
lr = LogisticRegression(max_iter=int(1e5), fit_intercept=False).fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]

# Train a Decision Tree model
dt = DecisionTreeClassifier(max_depth=7)
dt.fit(X_train, y_train)
dt_probs = dt.predict_proba(X_test)[:, 1]

# Hyperparameter tuning for Decision Tree using GridSearch
params_dt = {'max_depth': [2, 3, 4, 6], 'min_samples_leaf': [1, 1.5, 2]}
dt_cv = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params_dt, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
dt_cv.fit(X_train, y_train)
dt_cv_probs = dt_cv.predict_proba(X_test)[:, 1]

# Train a Random Forest model and make predictions
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# Plot ROC curves for models
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
dt_cv_auc = roc_auc_score(y_test, dt_cv_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

# ROC curve plotting
for fpr, tpr, label in [(roc_curve(y_test, ns_probs)[:2] + ('No Skill',)),
                        (roc_curve(y_test, lr_probs)[:2] + ('Logistic Regression',)),
                        (roc_curve(y_test, dt_probs)[:2] + ('Classification Tree',)),
                        (roc_curve(y_test, dt_cv_probs)[:2] + ('Tree with CV',)),
                        (roc_curve(y_test, rf_probs)[:2] + ('Random Forest',))]:
    plt.plot(fpr, tpr, label=label)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Print AUC scores
print('Random forest: ROC AUC=%.3f' % rf_auc)
print('Tree w/ cv: ROC AUC=%.3f' % dt_cv_auc)
print('Tree: ROC AUC=%.3f' % dt_auc)
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Logistic: ROC AUC=%.3f' % lr_auc)

# Print confusion matrix for random forest
print(confusion_matrix(y_test, rf.predict(X_test)))

# Calculate feature importances using permutation importance
rf_imp = permutation_importance(rf, X, y)
dt_imp = permutation_importance(dt, X, y)
dt_cv_imp = permutation_importance(dt_cv, X, y)
lr_imp = permutation_importance(lr, X, y)

# Get feature importance scores
importance_data = [
    ('RF', rf_imp.importances_mean),
    ('Tree', dt_imp.importances_mean),
    ('Tree w/ CV', dt_cv_imp.importances_mean),
    ('Logistic', lr_imp.importances_mean)
]

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['blue', 'red', 'orange', 'green']
for model_name, scores, color in zip(*zip(*importance_data), colors):
    indices = scores.argsort()[::-1]
    ax.barh(np.array(X.columns)[indices], np.array(scores)[indices], color=color, label=model_name)
ax.set_xlabel('Relative Importance')
ax.set_ylabel('Predictors')
ax.legend()
plt.show()