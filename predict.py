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
            break
except FileNotFoundError:
    sys.exit("[ERROR] FileNotFound, Training program should be launched before using the prediction program")

try:
    value = float(input("Enter the number of kilometers: "))
except ValueError:
    print("[ERROR] Unexpected value type")
    exit()
print(predict_value(theta1, theta0, value))