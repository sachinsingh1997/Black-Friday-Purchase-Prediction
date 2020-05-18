# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:46:02 2018

@author: sachin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#data preprocessing
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#NAN to mean value
test['Product_Category_2'].fillna(math.ceil(test['Product_Category_2'].mean()),inplace=True)
train['Product_Category_2'].fillna(math.ceil(train['Product_Category_2'].mean()),inplace=True)

#creating dummy columns of 1&0 form
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ['Gender','City_Category','Stay_In_Current_City_Years','Age']:
    train = create_dummies(train,column)
    test = create_dummies(test,column)
    
# dividing into dependent and indepensent variables
X=train.iloc[: ,[0,4,7,8,9,12,14,15,17,18,19,20,23,24,25,26]].values
y=train.iloc[:,11].values
X_test=test.iloc[:,[0,4,7,8,9,11,13,14,16,17,18,19,22,23,24,25]].values
df=pd.DataFrame(X)

#splitting data into test and train
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20,random_state=0)

#scttered plot
plt.scatter(X[:,2],y)
plt.show()
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(train_X, train_y)
regressor.feature_importances_
y_pred=regressor.predict(test_X)
y_test_pred=regressor.predict(X_test)
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
X_test_poly=poly_reg.fit_transform(X_test)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# corelation matrix plot
import matplotlib.pyplot as plt

plt.matshow(df.corr())
# predicting data of test
classifier.fit(all_X,all_y)
test_pred=classifier.predict(test[columns])

submission_df={"User_ID":test["User_ID"],"Product_ID":test['Product_ID'],"Purchase":y_test_pred}
submission=pd.DataFrame(submission_df)

submission.to_csv("submission.csv",index=False)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(test_X)
X_test = sc_X.transform(X_test)
