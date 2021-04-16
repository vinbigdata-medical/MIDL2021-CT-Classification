import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

"""This file read input as a .csv of F1 score after performing 
bootstrapping, then plot a 95% confidence interval graph with 
x-axis being number of slices taken from each scans and y-axis 
beign the F1-score of scan level prediction
"""


### PREPARE VARIABLES ###
bootstrap = pd.read_csv('bootstrap_168.csv')
key1 = 'Percentage of Slice for study level prediction'
data = {'Accuracy': [], key1: []}
keys = list(bootstrap.columns.values)
#########################

### PREPARE DIRECTORY ###
save_dir = 'CI_plot'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
#########################

for i, row in bootstrap.iterrows():
    for key in keys:
        data[key1].append(int(key))
        data['Accuracy'].append(row[key])  
df = pd.DataFrame(data)

ax = sns.lineplot(x=df[key1], y=df['Accuracy'], ci=95)
plt.savefig(f'{save_dir}/bootstrap.png')