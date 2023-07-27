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
prefix = "playoff"  

for column in playoffStats.columns:
    if column != "Player":
        playoffStats.rename(columns={column: prefix + column}, inplace=True)

        
playoffStats.to_csv('playoffStatsNBA.csv')
