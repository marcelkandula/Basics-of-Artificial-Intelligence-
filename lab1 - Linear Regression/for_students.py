import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()


# TODO: calculate closed-form solution
observation_matrix = np.hstack([np.ones(x_train.size).reshape(-1, 1), x_train.reshape(-1, 1)])
column_vector = y_train.reshape(-1, 1)

theta_best = np.linalg.inv(observation_matrix.T @ observation_matrix) @ observation_matrix.T @ column_vector
theta_best = theta_best.flatten()
print(f"Theta best for closed-from solution: {theta_best}")

# TODO: calculate error
mse_cfs = sum((theta_best[0] + theta_best[1] * x_train - y_train)**2)/x_train.size
print(f"MSE for closed-form solution: {mse_cfs}")


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)

x_train = (x_train - x_train_mean)/x_train_std
y_train = (y_train - y_train_mean)/y_train_std

x_test = (x_test - x_train_mean)/x_train_std
y_test = (y_test - y_train_mean)/y_train_std

# TODO: calculate theta using Batch Gradient Descent
theta_best = np.random.rand(2, 1)
learning_rate = 1e-3

observation_matrix = np.hstack([np.ones(x_train.size).reshape(-1, 1), x_train.reshape(-1, 1)])
column_vector = y_train.reshape(-1, 1)

mse = [0, 0]
old_mse = [1, 1]
eps = 1e-32
iterations = 0
while abs(old_mse[1] - mse[1]) > eps:
#for _ in range(int(1e5)):
    old_mse = mse
    mse = 2/(x_train.size) * observation_matrix.T @ (observation_matrix @ theta_best - column_vector)
    theta_best = theta_best - learning_rate * mse
    iterations += 1
theta_best = theta_best.flatten()
print(f"Theta best for Batch Gradient Descent after {iterations} iterations: {theta_best}")

# TODO: calculate error
mse_bgd = sum((theta_best[0] + theta_best[1] * x_train - y_train)**2)/x_train.size
print(f"MSE for Batch Gradient Descent: {mse_bgd}")

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()