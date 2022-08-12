# Multiple Linear Regression Preprocessing template

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing the dataset
dataset = pd.read_csv('org_data.csv')

# segregagting the independent values from the dependent values
# X, independent values and Y, dependent values
X = X2 = dataset.iloc[:,:-1].values
Y = Y2 = dataset.iloc[:, 4].values
 
#check for missing values
dataset[pd.notnull(dataset)] #no missing values identified


# Encoding Categorical data
#Encoding independent variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
ct = make_column_transformer((OneHotEncoder(),[3]), remainder='passthrough') #remaining columns won't be affected hence "passthrough"
X = ct.fit_transform(X)


#avoid dummy variable trap
X = X[:, 1:]

# spliting our data to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

#train the model to fit the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#predict our y values with x_test
y_pred = regressor.predict(X_test)

#mse to evaluate level of accuracy
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, y_pred, squared = False)

##optimising or improving performance of model using score method
regressor.score(X_train, Y_train)
regressor.score(X_test, Y_test) #csores are closely linked, thus an appropriate model of choice for this dataset situation


#optimising or improving performance of model using backward elimination
#feature Selection
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis = 1)

#ordinary least squares 
#where exog affects everything outside Y(dependent variable)
#where enog affects everything inside X(independent variable)
X_opt = X[:, [0,1,2,3,4,5]].astype(float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#after checking which row 3 has highest P > SL of 0.05
X_opt = X[:, [0,1,3,4,5]].astype(float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#after checking which row has highest P > SL of 0.05
X_opt = X[:, [0,3,4,5]].astype(float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#after checking which index position last but one has which is 4 highest P > SL of 0.05
X_opt = X[:, [0,3,5]].astype(float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#after checking which index position last but one has which is 5 highest P > SL of 0.05
X_opt = X[:, [0,3]].astype(float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#Part II
#Now use Research as the desired independent variable to give a good prediction
# X, independent value represents Research and Y, dependent values
X2 = dataset.iloc[:,:1].values
Y2 = dataset.iloc[:,4].values

# spliting our data to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

#train the model to fit the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#predict our y values with x_test
y_pred = regressor.predict(X_test)

#check performance of medel using score method
regressor.score(X_train, Y_train)
regressor.score(X_test, Y_test)  #looks pretty much same


#mse
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, y_pred, squared = False)