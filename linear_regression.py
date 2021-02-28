import numpy as np
import matplotlib.pyplot as plt
import csv

# Initialize datasets
# empty datasets
X = []
Y = []

# import values from food-profit
with open('01-food-profit.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    X.append(float(row['x']))
    Y.append(float(row['y']))

# Linear Regression Model
class LinearRegression:
  def train(self, X, Y):
    def hyp(x):
      return self.m * x + self.b

    n = len(X)
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    numer = 0
    denom = 0
    for i in range(n):
      numer += (X[i] - mean_x) * (Y[i] - mean_y)
      denom += (X[i] - mean_x) ** 2
    self.m = numer / denom
    self.b = mean_y - (self.m * mean_x)
    return hyp

  def test(self, hyp, x):
    print(f'Testing x={x}, predicts y={hyp(x)}')

# Training model
regression_model = LinearRegression()
hyp = regression_model.train(X, Y)
m = regression_model.m
b = regression_model.b

# Testing hypothesis
regression_model.test(hyp, 7.5)
regression_model.test(hyp, 5)
regression_model.test(hyp, 14)

# Plot
# plotting line
max_x = np.max(X) + 1
min_x = np.min(X) - 1
lin_x = np.linspace(min_x, max_x, 10)
lin_y = m * lin_x + b
plt.plot(lin_x, lin_y, c='#58b970', label='Regression Line')

# plotting points
plt.scatter(X, Y, s=10, label='Scatter points')

# show plot
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
