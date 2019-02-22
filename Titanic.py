# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:30:02 2019

@author: Shyam Parmar
"""

import numpy as np                # numpy is numerical python. Used for performing scientific calculaations
import pandas as pd               # pandas is used for data analysis
import seaborn as sns             # seaborn is a library for statistical plotting/ statistical data visualisation
import matplotlib.pyplot as plt   # for plotting
import math                       # to calculate basic math functions
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Import dataset
titanic_data = pd.read_csv('Titanic.csv')  

# Analyze data
sns.countplot(x="survived", data = titanic_data)
sns.countplot(x="survived", hue = "sex", data = titanic_data)
sns.countplot(x="survived", hue = "pclass", data = titanic_data)
titanic_data["age"].plot.hist()
titanic_data["fare"].plot.hist(figsize = (10,5))
sns.countplot(x="sibsp", data = titanic_data)  # Number of siblings or spouse

# Data Wrangling = Clean the data by removing NaN values and unnecessary columns in dataset
titanic_data.isnull()              # shows the nan values in all columns. True means nan 
titanic_data.isnull().sum()        # will print sum of number of nan values of all columns. 
sns.boxplot(x="pclass", y="age", data = titanic_data)  # Box plot to compare the age of passengers and the class they travel accordingly

# Data Cleaning
titanic_data.drop("cabin", axis =1, inplace = True)  #dropping or removing the cabin column as it has too much nan values and also this column is of no use to us
titanic_data.dropna(inplace = True)  #removes all nan or null values from dataset  

# Strings have to be converted into categorical variables. So we'll use dummy variables. With strings you can't predict anything so they've to be converted.

sex = pd.get_dummies(titanic_data['sex'], drop_first = True) #drop first column which assigns female value. As from male values only we can get whethter a person is male or not.0 is female & 1 is male
embarked = pd.get_dummies(titanic_data['embarked'], drop_first = True)
pcl = pd.get_dummies(titanic_data['pclass'], drop_first = True)

titanic_data = pd.concat([titanic_data, sex, embarked, pcl], axis = 1) #concatenate updated columns to the original data frame
titanic_data.drop(['sex','embarked','name','ticket','pclass'], axis = 1, inplace = True)

# Train and Test data

# First define dependent and independent variables as y and X respectively
X = titanic_data.drop("survived", axis = 1)  # Except survived all other variables will be taken as independent variables 
y = titanic_data["survived"]  # We need to predict whether the passenger survived or not so we take survived in y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
logmodel = LogisticRegression()   # Create an instance
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

classification_report(y_test, predictions)  # Can also use confusion_matrix

accuracy_score(y_test, predictions)  #check how accurate our model is or how accurate our results are
