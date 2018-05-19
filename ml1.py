# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns',60)
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.size']=24
from IPython.core.pylabtools import figsize
import seaborn as sns
sns.set(font_scale=2)
from sklearn.model_selection import train_test_split

os.chdir('E:\\Library\\learning\\programming')
os.listdir('.')

def gen_missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_perc = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_perc], axis = 1)
    mis_val_table = mis_val_table.rename(columns = {0:'Missing Values',1:'% of Total values'})
    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values('% of Total values', ascending=False)
    print('The dataset has ',str(df.shape[1]), 'columns \n', str(mis_val_table.shape[0]),'columns have missing values')
    return mis_val_table

xl = pd.ExcelFile(os.listdir('.')[0])
print(xl.sheet_names)
df = xl.parse('Information and Metrics')
df.info()

df = df.replace({'Not Available': np.nan})

for col in list(df.columns):
    if('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        df[col] = df[col].astype(float)
        
df.describe()
df.info()

df_missing = gen_missing_values(df)
missing_columns = list(df_missing[df_missing['% of Total values'] > 50].index)
print('%d columns removed'% len(missing_columns))
df.drop(missing_columns, axis=1, inplace=True)

figsize(8,8)

df = df.rename(columns={'ENERGY STAR Score':'score'})

#histogram of energy star score
plt.style.use('fivethirtyeight')
plt.hist(df['score'].dropna(), bins=100, edgecolor='k');
plt.xlabel('ccore');plt.ylabel('Number of buildings');
plt.title('Energy star score distribution')