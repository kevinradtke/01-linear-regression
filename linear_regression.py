import numpy as np
import matplotlib.pyplot as plt
import csv

# Import values from csv
def read_csv(X, Y, file_name):
  '''Reads a CSV with two float columns named "x" and "y" and fills X and Y lists'''
  with open(file_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      X.append(float(row['x']))
      Y.append(float(row['y']))

# Linear Regression Model
class LinearRegression:
  def fit(self, X, Y):
    '''Uses least square regression to find m and b'''
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
    print(f'fit results\t t0: {self.b}, t1: {self.m}')

  def predict(self, x):
    '''Predicts y value using fit results'''
    return self.m * x + self.b

  def calc_cost(self, X, Y, t0, t1):
    '''Calculates cost using predicted thetas'''
    m = len(X)
    cost = 0
    for i in range(m):
      prediction = t0 + t1 * X[i]
      cost += prediction - Y[i]
    return (1/2*m) * cost

  def grad_desc(self, X, Y, alpha=0.02, iterations=10000, err=0.001):
    '''
    Gradient descent and stores cur cost through epochs.
    Breaks cycle if cost converges given threshold err.
    '''
    m = len(X)
    cost_history = np.zeros(iterations)
    theta_0 = 1
    theta_1 = 1

    for it in range(iterations):
      sum_0 = 0
      sum_1 = 0
      for i in range(m):
        prediction = theta_0 + theta_1 * X[i]
        sum_0 += prediction - Y[i]
        sum_1 += (prediction - Y[i]) * X[i]
      new_theta_0 = theta_0 - (1/m) * alpha * sum_0
      new_theta_1 = theta_1 - (1/m) * alpha * sum_1
      theta_0 = new_theta_0
      theta_1 = new_theta_1
      print(f'epoch {it}\t t0: {theta_0}, t1: {theta_1}')
      cost_history[it] = self.calc_cost(X, Y, theta_0, theta_1)
      if (abs(cost_history[it] - cost_history[it-1])  < err):
        break
    return theta_0, theta_1, cost_history

def test_hypothesis(model, samples):
  '''Tests hypothesis with sample inputs'''
  for x in samples:
    print(f'Testing x={x}, predicts y={model.predict(x)}')

# Plots
# plotting scatter line
def plot_scatter_line(X, Y, m, b, title):
  '''Plots scatter coordinate points X and Y, and a line with slope m and intercept b'''
  min_x = np.min(X)
  max_x = np.max(X)
  lin_x = np.linspace(min_x, max_x)
  lin_y = m * lin_x + b
  plt.plot(lin_x, lin_y, c='#58b970', label='Regression Line')
  plt.scatter(X, Y, s=10, label='Scatter points')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(title)
  plt.legend()
  plt.show()

# plot cost history
def plot_cost(cost):
  '''Plots historical cost points'''
  plt.scatter(list(range(1, len(cost) + 1)), cost, s=10, label='Cost')
  plt.xlabel('epochs')
  plt.ylabel('cost')
  plt.title('Cost function')
  plt.legend()
  plt.show()

# Initialize datasets
X = []
Y = []
# file_name = '01-sea-temperature.csv'
file_name = '01-food-profit.csv'
read_csv(X, Y, file_name)

# Training model
model = LinearRegression()
t0, t1, cost_history = model.grad_desc(X, Y)
model.fit(X, Y)

# Testing model
test_hypothesis(model, [7.5, 5, 14, 20]) # testing random values

# Plotting Results
plot_scatter_line(X, Y, model.m, model.b, 'Least Square Regression')
plot_scatter_line(X, Y, t1, t0, 'Gradient Descent')
plot_cost(cost_history)
