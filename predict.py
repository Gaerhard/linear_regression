from sqlite3 import Row
import csv
import sys
import os

def predict_value(theta1, theta0, x):
    return (theta1 * x + theta0)

if (os.path.isfile("theta_values") == False):
    sys.exit("[ERROR] Dataset isn't a regular file or doesn't exist")
try:
    with open('theta_values', newline='') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            try:
                theta1 = float(row[0])
                theta0 = float(row[1])
            except ValueError:
                sys.exit("[ERROR] theta0 and theta1 should be numbers")
            except IndexError:
                sys.exit("[ERROR] theta1 and theta0 should be defined")
            break
    value = float(input("Enter the number of kilometers: "))
    print(predict_value(theta1, theta0, value))
except FileNotFoundError:
    sys.exit("[ERROR] FileNotFound, Training program should be launched before using the prediction program")
except ValueError:
    sys.exit("[ERROR] Unexpected value type")
except NameError:
    sys.exit("[ERROR] theta values aren't defined")
