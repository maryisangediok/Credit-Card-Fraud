# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 20:52:09 2022

@author: user
"""

import matplotlib.pyplot as plt
from dataset import get_df_original

dir_path = os.path.dirname(os.path.realpath(__file__))

#get data
df = get_df_original()

#select one location
df_syd = df[(df['Location'] == 'Sydney') &
 (df.index > '2015-01-01')]

rainfall = df_syd['Rainfall'].values
raintoday = df_syd['RainToday']\
 .map({'Yes': 1, 'No': 0}).values
fig, axs = plt.subplots(2)
axs[0].set_title('Rainfall')
axs[0].plot(rainfall)
axs[1].set_title('Rain Classification')
axs[1].bar(range(len(raintoday)), raintoday, color = 'red')
#plt.show()
plt.savefig(f'{dir_path}/visual/rainfallplot.png')
