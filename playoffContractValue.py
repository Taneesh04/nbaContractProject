import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from unidecode import unidecode

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
    playoffStats = pd.read_csv(r"C:\Users\TaneeshA\playoffStatsNBA.csv")
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
