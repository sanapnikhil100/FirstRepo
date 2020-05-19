import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import database
dataset = pd.read_csv("data1.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
plt.scatter(X, y)
plt.show()

X = np.array(X)
y = np.array(y)

regressor = LinearRegression()  
regressor.fit(X, y)

print(regressor)

#To retrieve the intercept:
print(f"Intercept: {regressor.intercept_}")

#For retrieving the slope:
print(f"Regression Coefficient: {regressor.coef_}")

#Equation of line
print(f"Equation of the line:\n y = {regressor.intercept_} + {regressor.coef_}x\n")

Xdata = dataset.iloc[:, :-1]
ydata = dataset.iloc[:, -1]
plt.scatter(Xdata, ydata)
#Xdata = np.array(Xdata.T)
#Xdata = Xdata[0]
y_vals = (regressor.intercept_)+ (regressor.coef_ )* Xdata
y_vals = np.array(y_vals)
#y_vals = y_vals[0]
plt.plot(Xdata, y_vals)
plt.show()


 
