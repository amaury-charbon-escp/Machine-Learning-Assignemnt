#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:50:08 2020

@author: amaurycharbon
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder as skl
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

train = pd.read_csv('/Users/amaurycharbon/Desktop/ESCP/Machine Learning with Python/ML Assignment/train.csv')
test = pd.read_csv('/Users/amaurycharbon/Desktop/ESCP/Machine Learning with Python/ML Assignment/test1.csv')

train.head(5)
test.head(5)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

#Creating a new variable to give more sense to data set
train['TimeDifference'] = train['purchaseTime'] - train['visitTime']
train.loc[train['TimeDifference'] < 0, 'Time_Difference'] = 0
train.loc[train['TimeDifference'] > 0, 'Time_Difference'] = train['TimeDifference']
train = train.drop(columns=['TimeDifference','purchaseTime'])

test['TimeDifference'] = test['purchaseTime'] - test['visitTime']
test.loc[test['TimeDifference'] < 0, 'Time_Difference'] = 0
test.loc[test['TimeDifference'] > 0, 'Time_Difference'] = test['TimeDifference']
test = test.drop(columns=['TimeDifference','purchaseTime'])

#Checking for missing values
train.isnull().sum()
test.isnull().sum()

#Change Dependent varibale to -1 and 0
train['label']= train['label'].replace(-1,0)
test['label']= test['label'].replace(-1,0)

#Droping certain variables deeemed useless or cases of multicollinearity
#and biases in model or too many unique values for C3
corr = train.corr()
train = train.drop(columns = ['id','C1','C10','visitTime','N4','N8','C3'])
test = test.drop(columns = ['id','C1','C10','visitTime','N4','N8','C3','label'])

#Encoding Data set
train.dtypes
train.astype({'hour': 'float64'}).dtypes
from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
train['C2'] = labelencoder.fit_transform(train['C2']).astype('int')
train['C4'] = labelencoder.fit_transform(train['C4']).astype('int')
train['C5'] = labelencoder.fit_transform(train['C5']).astype('int')
train['C6'] = labelencoder.fit_transform(train['C6']).astype('int')
train['C7'] = labelencoder.fit_transform(train['C7']).astype('int')
train['C8'] = labelencoder.fit_transform(train['C8']).astype('int')
train['C9'] = labelencoder.fit_transform(train['C9']).astype('int')
train['C11'] = labelencoder.fit_transform(train['C11']).astype('int')
train['C12'] = labelencoder.fit_transform(train['C12']).astype('int')

test['C2'] = labelencoder.fit_transform(test['C2']).astype('int')
test['C4'] = labelencoder.fit_transform(test['C4']).astype('int')
test['C5'] = labelencoder.fit_transform(test['C5']).astype('int')
test['C6'] = labelencoder.fit_transform(test['C6']).astype('int')
test['C7'] = labelencoder.fit_transform(test['C7']).astype('int')
test['C8'] = labelencoder.fit_transform(test['C8']).astype('int')
test['C9'] = labelencoder.fit_transform(test['C9']).astype('int')
test['C11'] = labelencoder.fit_transform(test['C11']).astype('int')
test['C12'] = labelencoder.fit_transform(test['C12']).astype('int')

#Balancing data set
train['label'].value_counts()
X_train = train.loc[:, train.columns !='label']
y_train = train.loc[:, train.columns == 'label']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state = 123)

X_train, y_train = os.fit_sample(X_train, y_train)
y_train['label'].value_counts()



#Applying a Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
model = LogisticRegression().fit(X_train, y_train)

test_table = pd.read_csv('/Users/amaurycharbon/Desktop/ESCP/Machine Learning with Python/ML Assignment/test1.csv')
test_final = test_table.iloc[:,1]

#Applying model to test set and getting probabilities
y_buy = model.predict(test)
y_pred = model.predict_proba(test)[:,1]

y_pred = pd.DataFrame(y_pred,columns = ['Probabilities'])
test_final = pd.DataFrame(test_final)

final_table = pd.concat([test_final, y_pred], axis = 1, join = 'inner')

final_table.to_csv('/Users/amaurycharbon/Desktop/export_final_table.csv', index = False, header = True )

