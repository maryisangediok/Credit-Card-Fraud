# libraries
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_df_original
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline


#file location
dir_path = os.path.dirname(os.path.realpath(__file__))

# Historical data
df = get_df_original()   
   
def preprocess(df):
    
    ###Data Cleaning
    
    #drop data with too many missing values
    #df = df.drop(['Sunshine','Evaporation','Cloud3pm','Cloud9am'], axis=1)
  
    ## Extract numerical features
    num_col = df.select_dtypes(include=np.number).columns.to_list()
    len(num_col) #12 +4
    
    # exrtract categorical features
    cat_col=df.select_dtypes(object).columns.tolist()
    len(cat_col) #6
        
    
    ###Visualize
    #we know that there are some different number of missing values to each features
    #depending on their dirstubtion we are going to replace with median or mean
    for i in num_col:
        fig, axs = plt.subplots(1,2,figsize=(15, 3))
        sns.histplot(df[i],bins=20, kde=True,ax=axs[0])
        sns.boxplot(df[i], ax = axs[1], color='#99befd', fliersize=1);
        #save figure to pc
        plt.savefig(f'{dir_path}/visual/distribution.png')
        plt.close()
 
    # fill missing values of normally-distributed columns with mean and skewed distribution with median
    df['Temp9am'] = df['Temp9am'].fillna(value = df['Temp9am'].mean())
    df['Temp3pm'] = df['Temp3pm'].fillna(value = df['Temp3pm'].mean())
    df['MinTemp'] = df['MinTemp'].fillna(value = df['MinTemp'].mean())
    df['MaxTemp'] = df['MaxTemp'].fillna(value = df['MaxTemp'].mean())
    df['Rainfall'] = df['Rainfall'].fillna(value = df['Rainfall'].mean())
    df['Humidity3pm'] = df['Humidity3pm'].fillna(value = df['Humidity3pm'].mean())
    
    median_values = df[num_col].median()
    df[num_col] = df[num_col].fillna(value=median_values)
    
    df['RainToday']=df['RainToday'].fillna(value = df['RainToday'].mode())
    df['RainTomorrow']=df['RainTomorrow'].fillna(value = df['RainTomorrow'].mode())
    
    # Convert categorized values to numerical values
    le = LabelEncoder()
    df[cat_col] =df[cat_col].astype('str').apply(le.fit_transform)
        
        
    # Check for multicollinearity
    # lets see the correlation between each variable by using heatmap
    fig, ax = plt.subplots(figsize=(20,10))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    sns.heatmap(df.corr(), annot=True, cmap="Reds", mask=mask, linewidth=0.5)
    #save figure to pc
    plt.savefig(f'{dir_path}/visual/multicollinearity.png')
    plt.close()    

    #define input values X, and target value y
    X = df.drop(['RainTomorrow'], axis = 1) 
    y = df['RainTomorrow'].values
    
    return X, y