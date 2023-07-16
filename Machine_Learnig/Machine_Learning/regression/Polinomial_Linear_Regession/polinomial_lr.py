import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#Reading the files : 
ds = pd.read_csv("Position_Salaries.csv")


X = ds.iloc[:,1:2].values
y = ds.iloc[:,2].values

# Bluffing detector : 
# Fitting Linear Regression in Dataset 
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(X, y)
#Fitting the Polynomial Regression into the data set : 
from sklearn.preprocessing import PolynomialFeatures
p_reg = PolynomialFeatures(degree=2)
X_poly = p_reg.fit_transform(X)
le2=LinearRegression()
le2.fit(X_poly,y)


