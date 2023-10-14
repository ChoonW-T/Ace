import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

pd.set_option('display.max_columns', None)

data = pd.read_excel('nba_player_data.xlsx')

data.isna().sum()
data.drop(columns=['RANK','EFF'], inplace=True)
data['season_start_year'] = data['Year'].str[:4].astype(int)

data['TEAM'].replace(to_replace=['NOP','NOH'], value='NO', inplace=True)
data['Season_type'].replace('Regular%20Season','RS', inplace=True)
data['PLAYER'] = data['PLAYER'].apply(lambda x: str(x).replace(".", ""))

rs_df = data[data['Season_type'] == 'RS']
playoffs_df = data[data['Season_type'] == 'RS']

total_cols = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']

data_per_min = data.groupby(['PLAYER','PLAYER_ID','Year'])[total_cols].sum().reset_index()

for col in data_per_min.columns[4:]:
    data_per_min[col] = data_per_min[col]/data_per_min['MIN']

data_per_min['FG%'] = data_per_min['FGM']/data_per_min['FGA']

data_per_min = data_per_min[data_per_min['MIN']>=50]
data_per_min.drop(columns='PLAYER_ID', inplace=True)

#fig = px.imshow(data_per_min.corr())
#fig.show()

fig = px.histogram(x=playoffs_df['MIN'], histnorm='percent')
fig.show()

def hist_data(df=rs_df, min_MIN=0, min_GP=0):
    df.loc[(df['MIN']>=min_MIN) & (df['GP']>=min_GP), 'MIN']/df.loc[(df['MIN']>=min_MIN) & (df['GP']>=min_GP), 'GP']

fig = go.Figure()

#fig.add_trace(go.Histogram(x=hist_data(rs_df,50,5), histnorm='percent', name='RS',xbin={'start':0,'end':46,'size':1}))
#fig.add_trace(go.Histogram(x=rs_df['MIN']/rs_df['GP'], histnorm='percent', name='RS'))
#fig.add_trace(go.Histogram(x=rs_df['MIN']/rs_df['GP'], histnorm='percent', name='RS',xbin={'start':0,'end':46,'size':1}))

#fig.add_trace(go.Histogram(x=hist_data(playoffs_df,50,5), histnorm='percent', name='RS',xbin={'start':0,'end':46,'size':1}))
#fig.add_trace(go.Histogram(x=playoffs_df['MIN']/rs_df['GP'], histnorm='percent', name='Playoffs'))

#fig.update_layout(barmode='overlay')
#fig.update_traces(opacity=0.5)
#fig.show()

#((hist_data(playoffs_df,5,1)>=12)&(hist_data(playoffs_df,5,1)<=34)).mean()

change_df = data.groupby('season_start_year')[total_cols].sum().reset_index()
change_df['POSS_est'] = change_df['FGA']-change_df['OREB']+change_df['TOV']+0.44*change_df['FTA']

change_df[list(change_df.columns[0:2])+['POSS_est']+list(change_df.columns[2:-1])]
change_df['FG%'] = change_df['FGM']/change_df['FGA']

change_per48_df = change_df.copy()

for col in change_per48_df.columns[2:18]:
    change_per48_df[col] = (change_per48_df[col]/change_per48_df['MIN']*48*5)

change_per48_df.drop(columns='MIN', inplace=True)
fig = go.Figure()

for col in change_per48_df[1:]:
    fig.add_trace(go.Scatter(x=change_per48_df['season_start_year'],y=change_per48_df[col], name=col))
fig.show()

rs_change_df = rs_df.groupby('season_start_year')[total_cols].sum().reset_index()
playoffs_change_df = playoffs_df.groupby('season_start_year')[total_cols].sum().reset_index()

for i in [rs_change_df,playoffs_change_df]:
    i['POSS_est'] = i['FGA']-i['OREB']+i['TOV']+0.44*i['FTA']
    i['POSS_per_48'] = (i['POSS_est']/i['MIN'])*48

    for col in total_cols:
        i[col] = 100*i[col]/i['POSS_est']

    i.drop(columns=['MIN','POSS_est'], inplace=True)

comp_change_df = round(100*(playoffs_change_df-rs_change_df)/rs_change_df,3)
comp_change_df['season_start_year'] = 2022