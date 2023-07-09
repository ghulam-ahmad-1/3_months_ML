import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
'''Working on iris Data set  |  Data Set From Kaggel '''
df = pd.read_csv("iris.csv")

# seprated the dependent  and Independent Varialbles : 
X = df.iloc[:,:4].values
y = df.iloc[:,-1].values

# Now Encoding the Catagorical Data :
le = LabelEncoder()
y=le.fit_transform(y)

# catagorical Data is Successfully Encoded .


