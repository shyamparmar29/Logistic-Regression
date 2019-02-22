# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 19:12:25 2019

@author: Shyam Parmar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris-data.csv')

#Removing all null values row
df = df.dropna(subset=['petal_width_cm'])

#Plot
sns.pairplot(df, hue='class', size=2.5)

df['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)

#Consider only two class 'Iris-setosa' and 'Iris-versicolor'. Dropping all other class
final_df = df[df['class'] != 'Iris-virginica']

#Outlier check
sns.pairplot(final_df, hue='class', size=2.5)

#From the above plot, sepal_width and sepal_length seems to have outliers. To confirm let's plot them seperately
final_df.hist(column = 'sepal_length_cm',bins=20, figsize=(10,5))

#It can be observed from the plot, that for 5 data points values are below 1 and they seem to be outliers. 
#So, these data points are considered to be in 'm' and are converted to 'cm'.
final_df.loc[final_df.sepal_length_cm < 1, ['sepal_length_cm']] = final_df['sepal_length_cm']*100

final_df = final_df.drop(final_df[(final_df['class'] == "Iris-setosa") & (final_df['sepal_width_cm'] < 2.5)].index)

# Plot again to check for successful removal of outliers
sns.pairplot(final_df, hue='class', size=2.5)

# Label Encoding
final_df['class'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True) 

# Model Consstruction
inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]], axis=1)

scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)

X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)

X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()

clf = LogisticRegression()
clf.fit(X_tr_arr, y_tr_arr)
pred = clf.predict(X_ts_arr)

print ('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))