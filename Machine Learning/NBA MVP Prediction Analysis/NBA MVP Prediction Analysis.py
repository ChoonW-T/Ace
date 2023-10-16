import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load the dataset
stats = pd.read_csv("player_mvp_stats.csv")

# Clean and preprocess the data
stats.drop(columns="Unnamed: 0", inplace=True)
stats['Player'] = stats['Player'].str.replace('.', '')
stats = stats.fillna(0)

# Encode categorical features
stats["NPos"] = stats["Pos"].astype("category").cat.codes
stats["Ntm"] = stats["Tm"].astype("category").cat.codes

# Normalize some stats
stats_ratios = stats[["PTS","AST","STL","BLK","3P","Year"]].groupby("Year").transform(lambda x: x/x.mean())
stats[["PTS_R","AST_R","STL_R","BLK_R","3P_R"]] = stats_ratios

# List of predictors
predictors = stats.columns.difference(['Share', 'Player', 'Tm', 'Pos'])

# Since we've encoded 'Tm' and 'Pos' as 'Ntm' and 'NPos', we should ensure only numeric columns are present.
predictors = [p for p in predictors if stats[p].dtype in ['int64', 'float64']]

# Function to calculate average precision
def find_ap(combination):
    actual = combination.sort_values("Share",ascending=False).head(5)
    predicted = combination.sort_values("predictions",ascending=False)
    ps = []
    found = 0
    seen = 1
    for index, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found/seen)
        seen += 1
    return sum(ps)/len(ps)

# Backtest function
def backtest(stats, model, years, predictors):
    aps = []
    all_combinations = []
    for year in years[5:]:
        train = stats[stats["Year"] < year].copy()
        test = stats[stats["Year"] == year].copy()
        model.fit(train[predictors], train["Share"])
        predictions = model.predict(test[predictors])
        test['predictions'] = predictions
        all_combinations.append(test)
        aps.append(find_ap(test))
    return sum(aps)/len(aps), aps, pd.concat(all_combinations)

models = {
    "Ridge Regression": Ridge(alpha=.1),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=50, random_state=1),
    "Support Vector Regression": SVR(kernel='linear'),
    "Linear Regression": LinearRegression()
}

results = {}

for model_name, model in models.items():
    mean_ap, aps, all_predictions = backtest(stats, model, list(range(1991,2022)), predictors)
    results[model_name] = {
        "mean_ap": mean_ap,
        "aps": aps,
        "all_predictions": all_predictions
    }

for model_name in results:
    print(f"Mean Average Precision (AP) of {model_name}: {results[model_name]['mean_ap']}")

# Access the trained Random Forest model from the models dictionary
rf = models["Random Forest Regressor"]

# Compute permutation importance
result = permutation_importance(rf, stats[predictors], stats["Share"], n_repeats=30, random_state=1)

# Sort the features by their importance
sorted_idx = result.importances_mean.argsort()

# Plot the feature importances
plt.figure(figsize=(10, 12))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=stats[predictors].columns[sorted_idx])
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()
