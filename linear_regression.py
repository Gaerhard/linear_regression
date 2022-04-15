import numpy
import matplotlib.pyplot as plot

data = numpy.genfromtxt('data1.csv', delimiter=',')[1:]
km = data[:,0]
price = data[:,1]

def plotData():
    plot.scatter(km, price)
    plot.xlabel("km")
    plot.ylabel("price")

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def denormalize(x, x_ref):
    return x * (x_ref.max() - x_ref.min()) + x_ref.min()

def lineFunction(x, theta0, theta1):
    return theta0 + x * theta1

def costFunction(xValues, yValues, theta0, theta1):
    return ((1 / (2 * len(xValues))) * sum(
        (lineFunction(x, theta0, theta1) - y) ** 2
         for x, y in zip(xValues, yValues)))


def theta0GradientDescent(xValues, yValues, theta0, theta1):
    return sum([lineFunction(x, theta0, theta1) - y
                for x, y in zip(xValues, yValues)]) / len(xValues)

def theta1GradientDescent(xValues, yValues, theta0, theta1):
    return sum([(lineFunction(x, theta0, theta1) - y) * x
                for x, y in zip(xValues, yValues)]) / len(xValues)


def gradientDescent(xValues, yValues, theta0, theta1, learningRate, iterations):
    for _ in range(iterations):
        nextTheta1 = theta1 - learningRate * theta1GradientDescent(xValues, yValues, theta0, theta1)
        nextTheta0 = theta0 - learningRate * theta0GradientDescent(xValues, yValues, theta0, theta1)
        theta0 = nextTheta0
        theta1 = nextTheta1
    return theta0, theta1

plotData()
theta1 = 0
theta0 = 0
learningRate = 0.1
nbIterations = 100000


xValues = (km)
yValues = (price)
# plot.show()
print(costFunction(xValues, yValues, theta0, theta1))
theta0, theta1 = gradientDescent(xValues, yValues, theta0, theta1, learningRate, nbIterations)
print(costFunction(xValues, yValues, theta0, theta1))
print(theta0)
print(theta1)
lineX = [xValues.min(), xValues.max()]
lineY = [lineFunction(x, theta0, theta1) for x in lineX]
print(lineY)
plot.plot(lineX, lineY, color='r')
plot.show()