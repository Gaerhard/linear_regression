from sqlite3 import DataError
from turtle import color
from numpy import float64, int64
import pandas
import matplotlib.pyplot as plot
import sys

def printHelp():
    print("This is a linear regression training program")
    print("     Usage: python|python3 ft_linear_regression.py [options]")
    print("     Options:")
    print("         -h  :  Help")
    print("         -p  :  Params")
    print("             Params option gives the possibility of choosing the number of iterations used to train the program")
    print("             and the value of the learning rate")
    print("         -g  :  Graph")
    print("         -a  :  Training accuracy")
    print("         -c  :  Cost evolution")
    print("         -r  :  Results of training")
    exit()

def parseArgs(args):
    graph = False
    setParams = False
    accuracy = False
    costEvolution = False
    trainingResults = False
    for arg in args:
        if arg == "-h":
            printHelp()
        elif arg == "-p":
            setParams = True
        elif arg == "-g":
            graph = True
        elif arg == "-a":
            accuracy = True
        elif arg == "-c":
            costEvolution = True
        elif arg == "-r":
            trainingResults = True
    return setParams, graph, accuracy, costEvolution, trainingResults

def f(theta1, x, theta0):
    return theta1*x+theta0

def costFunction(xData, yData, theta1, theta0):
    return sum([(f(theta1, x, theta0) - y)**2
        for x, y in zip(xData, yData)]) / 2 * xValues.size

def theta0GradientDescent(xValues, yValues, theta1, theta0):
    return sum([f(theta1, x, theta0) - y
                for x, y in zip(xValues, yValues)]) / xValues.size

def theta1GradientDescent(xValues, yValues, theta1, theta0):
    return sum([(f(theta1, x, theta0) - y) * x
                for x, y in zip(xValues, yValues)]) / xValues.size

def gradientDescent(nbIterations, theta1, theta0, learningRate, xValues, yValues, costEvolution):
    costData = []
    costXs = []
    if costEvolution:
        for i in range(10):
            costXs.append(int(nbIterations * i / 10))
        costXs.append(nbIterations - 1)
    for i in range(nbIterations):
        if costEvolution and i in costXs:
            costData.append(costFunction(xValues, yValues, theta1, theta0))
        nexttheta1 = theta1 - learningRate * theta1GradientDescent(xValues, yValues, theta1, theta0)
        nexttheta0 = theta0 - learningRate * theta0GradientDescent(xValues, yValues, theta1, theta0)
        theta1 = nexttheta1
        theta0 = nexttheta0
    return theta1, theta0, costData, costXs

def findPowerOfNumber(number):
    p = 1
    while number > 1:
        number = number / 10
        p = p * 10
    return p

def calcMean(values):
    return sum([value for value in values]) / values.size

def calcSumOfTotalSquares(yValues, yMean):
    return sum([(y - yMean)**2 for y in yValues])

def calcSumOfTotalPredictedSquares(xValues, yMean, theta1, theta0):
    return sum([(f(theta1, x, theta0) - yMean)**2 for x in xValues])

def calcCoefficientOfDetermination(yMean, xValues, yValues, theta1, theta0):
    sct = calcSumOfTotalSquares(yValues, yMean)
    sce = calcSumOfTotalPredictedSquares(xValues, yMean, theta1, theta0)
    try:
        return sce/sct
    except ZeroDivisionError:
        sys.exit("[ERROR] Dataset seems to be an equation of the type y=b so training accuracy cannot be determined")

setParams, graph, accuracy, costEvolution, trainingResults = parseArgs(sys.argv)
csvDataset = input("Choose the dataset you want to train: ")
try:
    dataFrame = pandas.read_csv(csvDataset)
except pandas.errors.EmptyDataError:
    sys.exit("[ERROR] Empty Dataset")
except FileNotFoundError:
    sys.exit("[ERROR] File not found")
except IsADirectoryError:
    sys.exit("[ERROR] Invalid Dataset")
except UnicodeDecodeError:
    sys.exit("[ERROR] Invalid Dataset")
if (dataFrame.isnull().values.any()):
    raise DataError('[ERROR] Invalid Dataset, there should be no null value')
xValues = dataFrame.iloc[0:dataFrame.size, 0]
yValues = dataFrame.iloc[0:dataFrame.size, 1]
if ((xValues.dtypes != float64 and xValues.dtypes != int64) or
    (yValues.dtypes != float64 and yValues.dtypes != int64)):
    raise TypeError('[ERROR] Only floats and integers are accepted')
theta1 = 0
theta0 = 0
if (setParams):
    try:
        nbIterations = int(input("Choose the number of iterations you want to realize: "))
        learningRate = float(input("What learning rate do you wish to use: "))
    except ValueError:
        sys.exit("[ERROR] The number of iterations should be an integer and the learning rate a float")
else:
    nbIterations = 100000
    learningRate = 0.1

maxPowerXValues = findPowerOfNumber(xValues.max())
maxPowerYValues = findPowerOfNumber(yValues.max())
if (maxPowerXValues >= maxPowerYValues):
    scale = maxPowerXValues
else:
    scale = maxPowerYValues
xValues = xValues / scale
yValues = yValues / scale
theta1, theta0, costData, costXs = gradientDescent(nbIterations, theta1, theta0, learningRate, xValues, yValues, costEvolution)
theta0 = theta0 * scale
xValues = xValues * scale
yValues = yValues * scale
with open('theta_values', "w+") as writer:
    writer.write(theta1.__str__() + "," + theta0.__str__())
if (graph):
    plot.axes().grid()
    plot.scatter(xValues,yValues)
    line_xs = [xValues.min(), xValues.max()]
    line_ys = [f(theta1, x, theta0) for x in line_xs]
    plot.plot(line_xs, line_ys, color='r')
    plot.show()
if (costEvolution):
    plot.axes().grid()
    plot.plot(costXs, costData, color='r')
    plot.show()
if (trainingResults):
    print("theta1 = ", theta1, "theta0 = ", theta0)
if (accuracy):
    print("Coefficient de determination = ", 
        calcCoefficientOfDetermination(calcMean(yValues), xValues, yValues, theta1, theta0))