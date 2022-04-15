import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('./data.csv', delimiter=',')[1:]
km = data[:, 0]
price = data[:, 1]

def plot_data():
    plt.scatter(km, price)
    plt.xlabel("km")
    plt.ylabel("price")

def h(x, theta1, theta0):
    return x * theta1 + theta0

def error_function(xs, ys, theta1, theta0):
    return (1 / (2 * len(xs))) * sum(
        [(h(x, theta1, theta0) - y) ** 2
         for x, y in zip(xs, ys)])

def theta0_partial(xs, ys, theta1, theta0):
    return sum([h(x, theta1, theta0) - y
                for x, y in zip(xs, ys)]) / len(xs)

def theta1_partial(xs, ys, theta1, theta0):
    return sum([(h(x, theta1, theta0) - y) * x
                for x, y in zip(xs, ys)]) / len(xs)


def gradient_descent(xs, ys, theta1, theta0, alpha, iterations):
    for _ in range(iterations):
        next_theta1 = theta1 - alpha * theta1_partial(xs, ys, theta1, theta0)
        next_theta0 = theta0 - alpha * theta0_partial(xs, ys, theta1, theta0)
        theta1 = next_theta1
        theta0 = next_theta0
    return theta1, theta0

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def denormalize(x, x_ref):
    return x * (x_ref.max() - x_ref.min()) + x_ref.min()

theta1 = -0.03
theta0 = 9000

xs = km
ys = price
X = normalize(xs)
Y = ys
print(error_function(xs, ys, theta1, theta0))
line_xs = [X.min(), X.max()]
line_ys = [h(x, theta1, theta0) for x in line_xs]

plt.scatter(X, Y)
plt.show()

plt.plot(line_xs, line_ys, color='r')

plot_data()
plt.show()