import pandas as pd
from bs4 import BeautifulSoup
import requests
from nba_api.stats.endpoints import leaguedashptdefend
from nba_api.stats.library.parameters import Season, SeasonTypeAllStar
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import re
from unidecode import unidecode

def create_driver():
    webdriver_service = Service('/path/to/chromedriver')
    driver = webdriver.Chrome(service=webdriver_service)
    return driver

def get_html(driver, url, button_id):
    driver.get(url)
    filter_button = driver.find_element(By.ID, button_id)
    filter_button.click()
    time.sleep(2)
    return driver.page_source

def extract_table_data(html, table_index, match_headers=True):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find_all('table')[table_index]
    headers = [th.text.strip() for th in table.find_all('tr')[1].find_all('th')]
    data = []
    for row in table.find_all('tr')[2:]:  # for every row in the table, excluding the first two
        cols = [col.text.strip() for col in row.find_all('td')]  # get the text in each column
        if match_headers and len(cols) != len(headers):
            continue
        data.append(cols)
    return pd.DataFrame(data, columns=headers)


def clean_df(df):
    for column in df.columns:    
        if column != "Name":
            df[column] = df[column].apply(lambda x: re.findall(r'\d+\.*\d*', str(x))[0] if pd.notnull(x) and re.findall(r'\d+\.*\d*', str(x)) else None)
    df['Name'] = df['Name'].str.split('\n', expand=True)[0]
    df['#'] = pd.to_numeric(df['#'], errors='coerce')
    df.dropna(subset=['#'], inplace=True)
    for column in df.columns:
        if column != 'Name':
            df[column] = pd.to_numeric(df[column], errors='coerce')
    df.reset_index(drop=True, inplace=True)
    df.fillna(0, inplace=True)
    df.rename(columns={"Name": "Player"}, inplace=True)
    return df

def get_playoff_defense():
    playoffDefense = leaguedashptdefend.LeagueDashPtDefend(
        season=Season.default,
        season_type_all_star=SeasonTypeAllStar.playoffs,
    ).league_dash_p_tdefend.get_data_frame()
    columns_to_drop = ['CLOSE_DEF_PERSON_ID', "PLAYER_LAST_TEAM_ID","PLAYER_LAST_TEAM_ABBREVIATION", "FREQ", "AGE", "G", "GP", "PLAYER_POSITION"]
    playoffDefense.drop(columns_to_drop, axis=1, inplace=True)
    playoffDefense.rename(columns={"PLAYER_NAME" : "Player"}, inplace=True)
    return playoffDefense

def get_basketball_reference_data(url):
    r = requests.get(url)
    c = r.content
    soup = BeautifulSoup(c, 'html.parser')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table), header=0)[0]
    df = df.loc[df['Rk'] != 'Rk']  
    numeric_columns = df.columns.drop(['Player', 'Tm', 'Pos']) 
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    columns_to_drop = ['Rk', 'Pos', 'Age', 'G']
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df
driver = create_driver()

playoffDefense = get_playoff_defense()

url_per_game = 'https://www.basketball-reference.com/playoffs/NBA_2023_per_game.html'
url_advanced = 'https://www.basketball-reference.com/playoffs/NBA_2023_advanced.html'
df_per_game = get_basketball_reference_data(url_per_game)
df_advanced = get_basketball_reference_data(url_advanced)
columns_to_drop = ['MP']
df_advanced.drop(columns_to_drop, axis=1, inplace=True)


url = "https://craftednba.com/playoff-stats"

advanced_html = get_html(driver, url, 'advanced')
advanced = extract_table_data(advanced_html, 1)
advanced = clean_df(advanced)
advanced.drop(['TS%', "ORB%", "rORB%", "DRB%", "rDRB%", "STL%", "BLK%", "TOV%", "MP", "USG%", "3PAr", "FTr"], axis=1, inplace=True)

defenseIMP_html = get_html(driver, url, 'plusminus')
defenseIMP = extract_table_data(defenseIMP_html, 2, match_headers=True)

defenseIMP = clean_df(defenseIMP)
defenseIMP.drop(['Age', 'MP', 'OBPM', 'DBPM', 'BPM', 'VORP'], axis=1, inplace=True)

driver.quit()

playoffStats = pd.merge(df_per_game, df_advanced, on=['Player', 'Tm'])


playoffStats["Player"] = playoffStats["Player"].apply(unidecode)


playoffStats = playoffStats.merge(playoffDefense, on="Player")
defenseIMP = defenseIMP.merge(advanced, on="Player")

playoffStats['Player'] = playoffStats['Player'].str.lower().str.strip()
defenseIMP['Player'] = defenseIMP['Player'].str.lower().str.strip()
playoffStats = playoffStats.merge(defenseIMP, on='Player', how='inner')
prefix = "playoff"  # Specify your desired prefix

for column in playoffStats.columns:
    if column != "Player":
        playoffStats.rename(columns={column: prefix + column}, inplace=True)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

def power_regression(df, input_column, output_column, initial_guess, benchmark_column, diff_column, threshold, salary_threshold=0, adjustment=0, title=''):
    # Create a copy of the DataFrame to use for fitting the model
    df_model = df.copy()
    
    # Apply salary_threshold filter if specified
    if salary_threshold > 0:
        df_model = df_model[df_model[output_column] >= salary_threshold]

    # Fit the model to the filtered DataFrame
    p , e = curve_fit(power_func, df_model[input_column].values, df_model[output_column].values, p0=initial_guess)

    # Apply the model to the original DataFrame
    df[benchmark_column] = power_func(df[input_column].values, *p)
    df.loc[df[input_column] > threshold, benchmark_column] += adjustment
    df[diff_column] = df[benchmark_column] - df[output_column]
    
    plt.scatter(df[input_column], df[output_column], color='red')
    plt.plot(sorted(df[input_column]), sorted(df[benchmark_column]), color='blue')
    plt.title(title)
    plt.xlabel(input_column)
    plt.ylabel(output_column)
    plt.show()

    
def power_func(x, a, b):
    return a * np.power(x, b)
    
def load_data():
    seasonPay = pd.read_csv(r"C:\Users\TaneeshA\OneDrive\Desktop\NBA Contact Value Project\DataSets\NBA2023Payroll.csv")
    seasonPay['Player'] = seasonPay['Player'].str.lower().str.strip()
    seasonPay["Player"] = seasonPay["Player"].apply(unidecode)
    playoffValue = playoffStats.merge(seasonPay, on="Player")
    playoffValue.fillna(0, inplace=True)
    playoffValue["playoffSTOCKS"] = playoffValue["playoffSTL"] + playoffValue["playoffBLK"]
    return playoffValue
    
def regression(df, model, input_column, output_column, benchmark_column, diff_column, title):
    model.fit(df[[input_column]], df[output_column])
    df[benchmark_column] = model.predict(df[[input_column]])
    df[diff_column] = df[benchmark_column] - df[output_column]
    plt.scatter(df[input_column], df[output_column], color='red')
    plt.plot(sorted(df[input_column]), sorted(df[benchmark_column]), color='blue')
    plt.title(title)
    plt.xlabel(input_column)
    plt.ylabel(output_column)
    plt.show()
    return df[diff_column]
    
def quantile_regression(df, input_column, output_column, benchmark_column, diff_column, mask_threshold=0, color1='red', color2='blue', title=''):
    X = df[input_column]
    y = df[output_column]
    X_const = sm.add_constant(X)
    mask = y > mask_threshold
    X1, y1 = X_const[mask], y[mask]  
    X2, y2 = X_const[~mask], y[~mask]  
    mod1 = smf.quantreg('y1 ~ X1', df[mask])
    res1 = mod1.fit(q=.50)
    mod2 = smf.quantreg('y2 ~ X2', df[~mask])
    res2 = mod2.fit(q=.50)
    df.loc[mask, benchmark_column] = res1.predict(X1)
    df.loc[~mask, benchmark_column] = res2.predict(X2)
    df[diff_column] = df[benchmark_column] - df[output_column]
    plt.scatter(X[mask], y[mask], color = color1)
    plt.scatter(X[~mask], y[~mask], color = color2)
    plt.plot(X[mask], res1.predict(X1), color = color1)
    plt.plot(X[~mask], res2.predict(X2), color = color2)
    plt.title(title)
    plt.xlabel(input_column)
    plt.ylabel(output_column)
    plt.show()


playoffValue = load_data()

power_regression(playoffValue, 'playoffMP', '2022-23', [1, 14], 'minutesBenchmark', 'minutesDifferentVar', 30, adjustment=12000000, title='Predicting Salary based on Playoff Minutes Played')
power_regression(playoffValue, 'playoffUSG%', '2022-23', [1, 12], 'expectedUSG', 'USGDifference', 1e7, salary_threshold = 16e6, title = 'Predicting Salary based on Playoff Usage Rate')


model = SGDRegressor(penalty='elasticnet', random_state=45)
regression(playoffValue, model, 'playoffPTS', '2022-23', 'PPGBenchmark', 'pointsDifference', 'ElasticNet Regression')

df_model = playoffValue[playoffValue['2022-23'] >= 1e6]
model = LinearRegression()
playoffValue['StockDifference'] = regression(df_model, model, 'playoffSTOCKS', '2022-23', 'expectedSTOCKS', 'StockDifference', 'Predicting Salary based on Playoff Stocks (Steals + Blocks)')


playoffValue['defADJ'] = playoffValue['playoffCDPM'] + playoffValue['playoffDRPTR'] + playoffValue['playoffBPM']
quantile_regression(playoffValue, 'defADJ', '2022-23', 'expectedDef', 'defenseCalc', mask_threshold=2e7, title = 'Salary Prediciton based on Defensive Analytics')

playoffValue['offADJ'] = playoffValue['playoffSQ'] * 12 + playoffValue['playoffLOAD'] * 3 + playoffValue['playoffORPTR'] * 10 
quantile_regression(playoffValue, 'offADJ', '2022-23', 'offBench', 'offDifferentVar', mask_threshold = 1e7, title='Salary Prediciton based on Offensive Analytics')


playoffValue['overallExpected'] = playoffValue['defenseCalc'] * 2 + playoffValue['StockDifference'] * 1.5 + playoffValue['minutesDifferentVar'] * 3 + playoffValue['playoffUSG%'] * 0.4 + playoffValue['pointsDifference'] * 1.5 + playoffValue['offADJ'] * 2.6
playoffValue['overallExpected'] /= 11
playoffValue.to_csv(r"C:\Users\TaneeshA\OneDrive\Desktop\NBA Contact Value Project\DataSets\NBATest1.csv")



import plotly.express as px

df = pd.read_csv(r"C:\Users\TaneeshA\OneDrive\Desktop\NBA Contact Value Project\DataSets\NBATest1.csv")
df.sort_values('overallExpected', ascending=False, inplace=True)


fig = px.scatter(df, x='Player', y='overallExpected', hover_data=['Player'])

fig.update_layout(title='NBA Players vs Overall Expected Values',
                  xaxis_title='Player',
                  yaxis_title='Overall Expected Value')

fig.show()
