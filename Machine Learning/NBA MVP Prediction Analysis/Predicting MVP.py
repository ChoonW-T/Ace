import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

stats = pd.read_csv("player_mvp_stats.csv")

del stats["Unnamed: 0"]
pd.isnull(stats).sum()
stats['Player'] = stats['Player'].str.replace('.','')

stats[pd.isnull(stats["3P%"])][["Player","3PA"]]
stats[pd.isnull(stats["FT%"])][["Player","FTA"]]

predictors = stats.columns

stats = stats.fillna(0)

train = stats[stats["Year"] < 2021]
test = stats[stats["Year"] == 2021]

reg = Ridge(alpha=.1)
reg.fit(train[predictors], train["Share"])

predictions = reg.predict(test[predictors])
predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)

combination = pd.concat([test[["Player","Share"]], predictions], axis = 1)

combination.sort_values("Share", ascending=False).head(10)

mean_squared_error(combination["Share"], combination["predictions"])

actual = combination.sort_values("Share", ascending = False)
combination["Rank"] = list(range(1,combination.shape[0]+1))

combination = combination.sort_values("predictions",ascending = False)
combination["Predicted Rank"] = list(range(1, combination.shape[0]+1))

combination.head(10)

def find_ap(combination):
    actual = combination.sort_values("Share",ascending=False).head(5)
    predicted = combination.sort_values("predictions",ascending=False)
    ps = []
    found = 0
    seen = 1
    for index, row in predicted.iterrows():
        if row["Player"] in actual["Player"]:
            found += 1
            ps.append(found/seen)
        seen += 1
    return ps

find_ap(combination)

years = list(range(1991,2022))

aps = []
all_predictions = []

for year in years[5:]:
    train = stats[stats["Year"]< year]
    test = stats[stats["Year"] == year]
    reg.fit(train[predictors],train["Share"])
    predictions = reg.predict(test[predictors])
    predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
    combination = pd.concat([test[["Player","Share"]], predictions], axis = 1)
    all_predictions.append((combination))
    aps.append(find_ap(combination))

sum(aps)/len(aps)

def add_ranks(combination):
    combination = combination.sort_values("Share", ascending = False)
    combination["Rank"] = list(range(1,combination.shape(0)+1))
    combination = combination.sort_values("predictions", ascending=False)
    combination["Predicted Rank"] = list(range(1,combination.shape(0)+1))
    combination["Diff"] = combination["Rank"]-combination["Predicted Rank"]
    return combination

ranking = add_ranks(all_predictions[1])
ranking[ranking["Rank"]<6].sort_values("Diff", ascending=False)

def backtest(stats, model, year, predictors):
    aps = []
    all_predictions = []
    for year in years[5:]:
        train = stats[stats["Year"]< year]
        test = stats[stats["Year"]== year]
        model.fit(train[predictors],train["Share"])
        predictions = reg.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns["predictions"], index=test.index)
        combination = pd.concat([test[["Player","Share"]], predictions], axis= 1)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps)/len(aps), aps, pd.concat(all_predictions)

mean_ap, aps, all_predictions = backtest(stats, reg, years[5:], predictors)

all_predictions[all_predictions["Rank"]<= 5].sort_values("Diff")/head(10)

pd.concat([pd.Series(reg.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False)

stat_ratios = stats[["PTS","AST","STL","BLK","3P","Year"]].groupby("Year").apply(lambda x: x/x.mean())

stats[["PTS_R","AST_R","STL_R","BLK_R","3P_R"]] = stat_ratios[["PTS","AST","STL","BLK","3P"]]
predictors += ["PTS_R","AST_R","STL_R","BLK_R","3P_R"]

stats["NPos"] = stats["Pos"].astype("category").cat.codes
stats["Ntm"] = stats["Tm"].astype("category").cat.codes

rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)
mean_ap_rf, aps_rf, all_predictions_rf = backtest(stats, rf, years[28:], predictors)

