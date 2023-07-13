import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# Reading the Data : 
ds = pd.read_csv("Salary_Data.csv")

X = ds.iloc[:,:-1].values # independent Variable 
y = ds.iloc[:,-1].values # Dependent Variable 

# Spliting the Data . 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Now Traing the Model . 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
score =reg.score(X_test, y_test)

# Predicting the test data .
y_pred = reg.predict(X_test)

# Visualizing the train Data set 
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, reg.predict(X_train), color = 'red')
plt.xlabel("Experience")
plt.ylabel("Salraries")
plt.show()

# Visualizing the Test data set .
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, reg.predict(X_train), color = 'red')
plt.xlabel("Experience")
plt.ylabel("Salraries")
plt.show()