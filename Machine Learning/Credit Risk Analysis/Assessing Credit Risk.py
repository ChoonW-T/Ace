import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# read the csv data using pd.read_csv function, define the first column as the index
df = pd.read_csv('Data.csv', index_col = 0)
print(df.shape)
df.head(10)

frequency = df['loan_status'].value_counts().idxmax()

df['loan_status'].fillna(value = frequency, inplace = True)

df_backup = df.copy()

check = df.isna().sum()

df = df.loc[:, df.isnull().sum() <= 10]
df.dropna()

df["loan_dummy"] = np.where(df["loan_status"] == "Charged Off", 1, 0)

# create a bar chart
df["loan_dummy"].value_counts().plot(kind='bar')

# add labels and title
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.title('Distribution of loan status')

# display the plot
plt.show()

grade_order = ['A', 'B', 'C', 'D', 'E', 'F','G']

sns.barplot(data=df, y='loan_amnt', x='grade',order=grade_order)
#sns.barplot(data=df, y='int_rate', x='grade',order=grade_order)
plt.show()

y = df['loan_dummy'].values
X = df[['loan_amnt', 'int_rate','annual_inc','total_pymnt','installment','total_rec_int','last_pymnt_amnt']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)


lr = LogisticRegression(max_iter=int(1e5), fit_intercept = False).fit(X_train, y_train)
yhat = lr.predict(X_test)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# generate the predicted probabilities from the logistic regression
lr_probs = lr.predict_proba(X_test)[:,1]

dt = DecisionTreeClassifier(max_depth = 7)
dt.fit(X_train, y_train)

dtree = DecisionTreeClassifier(ccp_alpha=0)
# ccp_alpha = 0 means there is not cost complexity pruning

# generate the predicted probabilities
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)

params_dt = {'max_depth': [2,3,4,6],
             'min_samples_leaf': [1, 1.5, 2]}

# Instantiate grid_dt
dt_cv = GridSearchCV(estimator=dtree,
                       param_grid=params_dt,
                       scoring='accuracy',
                       cv=5,
                       verbose=1,
                       n_jobs=-1)

dt_cv.fit(X_train, y_train)

# First create the base model to tune
rf = RandomForestClassifier(n_estimators=10)

# Fit the random search model
rf.fit(X_train, y_train)

y_dt_pred = dt.predict(X_test)
dt_probs = dt.predict_proba(X_test)[:,1]

rf_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:,1]

y_dt_cv_pred = dt_cv.predict(X_test)
dt_cv_probs = dt_cv.predict_proba(X_test)[:,1]

# generate the predicted probabilities from the logistic regression
dt_cv_fpr, dt_cv_tpr, _ = roc_curve(y_test, dt_cv_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
dt_cv_auc = roc_auc_score(y_test, dt_cv_probs)

# plot the roc curve for the model
plt.plot(rf_fpr, rf_tpr, linestyle='-', color='y',label='Random Forest')
plt.plot(dt_cv_fpr, dt_cv_tpr, linestyle='-',color='c', label='Classification Tree with CV')
plt.plot(dt_fpr, dt_tpr, linestyle='-', color='r',label='Classification Tree')
plt.plot(ns_fpr, ns_tpr, linestyle='--', color='b',label='No Skill')
plt.plot(lr_fpr, lr_tpr, linestyle='-', color='g',label='Logistic Regression')

# axis labelsS
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the legend
plt.legend()

# show the plot
plt.show()

# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)

# summarize scores
print('Random forest: ROC AUC=%.3f' % (rf_auc))
print('Tree w/ cv: ROC AUC=%.3f' % (dt_cv_auc))
print('Tree: ROC AUC=%.3f' % (dt_auc))
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

print(metrics.confusion_matrix(y_test, rf_pred))

from sklearn.inspection import permutation_importance
# Calculate permutation importance
rf_imp = permutation_importance(rf, X, y)
dt_imp = permutation_importance(dt, X, y)
dt_cv_imp = permutation_importance(dt_cv, X, y)
lr_imp = permutation_importance(lr, X, y)

# Get feature importance scores
rf_scores = rf_imp.importances_mean
dt_scores = dt_imp.importances_mean
dt_cv_scores = dt_cv_imp.importances_mean
lr_scores = lr_imp.importances_mean

# Get feature names
feature_names = X.columns

# Sort feature importances in descending order
sorted_indices_rf = rf_scores.argsort()[::-1]
sorted_indices_dt = dt_scores.argsort()[::-1]
sorted_indices_dt_cv = dt_cv_scores.argsort()[::-1]
sorted_indices_lr = lr_scores.argsort()[::-1]

# Print feature importances in descending order
for idx in sorted_indices_rf:
    print(f"{feature_names[idx]}: {rf_scores[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x=feature_names, y=rf_scores, ax=ax, color='blue')
sns.barplot(x=feature_names, y=dt_scores, ax=ax, color='red')
sns.barplot(x=feature_names, y=dt_cv_scores, ax=ax, color='orange')
sns.barplot(x=feature_names, y=lr_scores, ax=ax, color='green')

ax.set_xlabel('Predictors')
ax.set_ylabel('Importance')

blue_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', edgecolor='none')
red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', edgecolor='none')
orange_patch = plt.Rectangle((0, 0), 1, 1, fc='orange', edgecolor='none')
green_patch = plt.Rectangle((0, 0), 1, 1, fc='green', edgecolor='none')

ax.legend([blue_patch, red_patch, orange_patch, green_patch], ["RF", "Tree", "Tree w/ CV","Logistic"])

# set plot title and axis labels
plt.show()

sorted_xrf = np.take(feature_names, sorted_indices_rf)
sorted_yrf = np.take(rf_scores, sorted_indices_rf)

sorted_xdt = np.take(feature_names, sorted_indices_dt)
sorted_ydt = np.take(dt_scores, sorted_indices_dt)

sorted_xdtcv = np.take(feature_names, sorted_indices_dt_cv)
sorted_ydtcv = np.take(abs(dt_cv_scores), sorted_indices_dt_cv)

sorted_xlr = np.take(feature_names, sorted_indices_lr)
sorted_ylr = np.take(abs(lr_scores), sorted_indices_lr)

#plt.barh(sorted_xdt, sorted_ydt)
#plt.barh(sorted_xdtcv, sorted_ydtcv)
#plt.barh(sorted_xrf, sorted_yrf)
#plt.barh(sorted_xlr, sorted_ylr)

plt.title("Random Forest")
plt.ylabel('Predictors')
plt.xlabel('Relative Importance')