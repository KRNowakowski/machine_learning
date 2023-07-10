import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
This program presents a simple linear regression model without data validation, in which we find 
the relationship between the age of the dog and the number of visits to the vet.
"""

df = pd.read_csv('data.csv', delimiter=",")


X = df.values[:, :-1]
y = df.values[:, -1]


line_fitting = LinearRegression().fit(X, y)


m = line_fitting.coef_.flatten()
b = line_fitting.intercept_.flatten()
print(f'm-factor: {m}')
print(f'b-factor: {b}')


plt.plot(X, y, 'o')
plt.plot(X, m*X+b)
plt.show()
