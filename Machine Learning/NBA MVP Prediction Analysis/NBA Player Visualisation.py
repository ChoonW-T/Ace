import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data.drop(columns=['RANK','EFF'], inplace=True)
    data['season_start_year'] = data['Year'].str[:4].astype(int)
    data['TEAM'].replace(to_replace=['NOP','NOH'], value='NO', inplace=True)
    data['Season_type'].replace('Regular%20Season','RS', inplace=True)
    data['PLAYER'] = data['PLAYER'].apply(lambda x: str(x).replace(".", ""))
    return data

def get_data_per_minute(data, total_cols):
    data_per_min = data.groupby(['PLAYER','PLAYER_ID','Year'])[total_cols].sum().reset_index()
    for col in data_per_min.columns[4:]:
        data_per_min[col] = data_per_min[col]/data_per_min['MIN']
    data_per_min['FG%'] = data_per_min['FGM']/data_per_min['FGA']
    data_per_min = data_per_min[data_per_min['MIN'] >= 50]
    data_per_min.drop(columns='PLAYER_ID', inplace=True)
    return data_per_min

def plot_histogram(df, column):
    fig = px.histogram(x=df[column], histnorm='percent')
    fig.show()

def get_change_per_season(data, total_cols):
    change_df = data.groupby('season_start_year')[total_cols].sum().reset_index()
    change_df['POSS_est'] = change_df['FGA'] - change_df['OREB'] + change_df['TOV'] + 0.44*change_df['FTA']
    change_df['FG%'] = change_df['FGM']/change_df['FGA']
    return change_df

def plot_metrics_over_years(change_df):
    fig = go.Figure()
    for col in change_df.columns[1:]:
        fig.add_trace(go.Scatter(x=change_df['season_start_year'], y=change_df[col], name=col))
    fig.show()

# Load and preprocess data
data = load_and_preprocess_data('nba_player_data.xlsx')
rs_df = data[data['Season_type'] == 'RS']
playoffs_df = data[data['Season_type'] != 'RS']

# Define columns to process
total_cols = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']

# Get per-minute stats and plot correlation
data_per_min = get_data_per_minute(data, total_cols)
# Example visualization
plot_histogram(playoffs_df, 'MIN')

# Get change per season and plot metrics over years
change_df = get_change_per_season(data, total_cols)
plot_metrics_over_years(change_df)