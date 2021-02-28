import numpy as np
import matplotlib.pyplot as plt
import csv

# init empty datasets
X = []
Y = []

# import values from food-profit
with open('01-food-profit.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    X.append(float(row['x']))
    Y.append(float(row['y']))

def linear_regression_train(X, Y):
  def hyp(x):
    return m * x + b

  n = len(X)
  mean_x = np.mean(X)
  mean_y = np.mean(Y)

  numer = 0
  denom = 0
  for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
  m = numer / denom
  b = mean_y - (m * mean_x)
  return hyp, m, b

def linear_regression_test(hyp, x):
  print(f'Testing x={x}, predicts y={hyp(x)}')

model = linear_regression_train(X, Y)
hyp = model[0]
m = model[1]
b = model[2]

linear_regression_test(hyp, 7)
linear_regression_test(hyp, 2)
linear_regression_test(hyp, 14)

# plot

# plotting line
max_x = np.max(X) + 1
min_x = np.min(X) - 1
lin_x = np.linspace(min_x, max_x, 10)
lin_y = m * lin_x + b
plt.plot(lin_x, lin_y, c='#58b970', label='Regression Line')

# plotting points
plt.scatter(X, Y, s=10, label='Scatter points')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
