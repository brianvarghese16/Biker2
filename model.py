# -*- coding: utf-8 -*-
!pip install numpy
!pip install scikit-learn
!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install pywin32
import numpy as np
import os 
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv(r'D:\BIKE\AdvWorksCustsConsolidated.csv')

df = pd.get_dummies(df, columns=["Education","Occupation","Gender","MaritalStatus","CountryRegionName"], drop_first = True)
df.columns

feature_df = df[['NumberCarsOwned',
       'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome',
       'AveMonthSpend','Age',
       'Education_Graduate Degree', 'Education_High School',
       'Education_Partial College', 'Education_Partial High School',
       'Occupation_Management', 'Occupation_Manual',
       'Occupation_Professional', 'Occupation_Skilled Manual',
       'Gender_M', 'MaritalStatus_S',
       'CountryRegionName_Canada',
       'CountryRegionName_France', 'CountryRegionName_Germany',
       'CountryRegionName_United Kingdom', 'CountryRegionName_United States']]
       
       
feature_df.rename(columns = {'Education_Graduate Degree':'Education_Graduate_Degree', 'Education_High School':'Education_High_School', 'Education_Partial College': 'Education_Partial_College', 'Education_Partial High School':'Education_Partial_High_School', 'Occupation_Skilled Manual':'Occupation_Skilled_Manual', 'CountryRegionName_United Kingdom':'CountryRegionName_United_Kingdom', 'CountryRegionName_United States':'CountryRegionName_United_States'}, inplace=True)

X = np.asarray(feature_df)
y = np.asarray(df["BikeBuyer"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
dt_model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes = 25, random_state=10)

dt_model.fit(X_train, y_train)
dt_model.score(X_train, y_train)
y_pred = dt_model.predict(X_test)
print(classification_report(y_test, y_pred))

file = open("decision_tree_model.pkl", "wb")

pickle.dump(dt_model, file)