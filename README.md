# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select R&D Spend and Profit variables.
2. Initialize theta values and compute the initial cost function.
3. Apply gradient descent to update theta and minimize error.
4. Plot the cost graph and regression line for prediction. 
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: THULASI J
RegisterNumber: 212224250020
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\k\Downloads\50_Startups.csv")

# Select independent and dependent variables
# Using R&D Spend -> Profit
x_data = data['R&D Spend'].values
y_data = data['Profit'].values

# Scatter Plot
plt.scatter(x_data, y_data)

plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

plt.show()

# Cost Function
def computecost(x, y, theta):
    m = len(y)

    h = x.dot(theta)

    square_err = (h - y) ** 2

    cost = (1 / (2 * m)) * np.sum(square_err)

    return cost


# Prepare X matrix
m = len(x_data)

x = np.append(np.ones((m, 1)),
              x_data.reshape(m, 1),
              axis=1)

y = y_data.reshape(m, 1)

# Initialize theta
theta = np.zeros((2, 1))

# Initial Cost
err = computecost(x, y, theta)

print("Initial Cost =", err)


# Gradient Descent Function
def gradientDescent(x, y, theta, alpha, num_iters):

    m = len(y)

    j_history = []

    for i in range(num_iters):

        predictions = x.dot(theta)

        error = np.dot(x.transpose(), (predictions - y))

        descent = alpha * (1 / m) * error

        theta = theta - descent

        j_history.append(computecost(x, y, theta))

    return theta, j_history


# Train Model
theta, j_history = gradientDescent(x, y, theta, 0.0000000001, 1500)

# Print Equation
print("h(x) =",
      round(theta[0, 0], 2),
      "+",
      round(theta[1, 0], 8),
      "x")

# Cost Function Graph
plt.plot(j_history)

plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function using Gradient Descent")

plt.show()

# Regression Line
plt.scatter(x_data, y_data)

x_value = np.linspace(min(x_data), max(x_data), 100)

y_value = theta[0, 0] + theta[1, 0] * x_value

plt.plot(x_value, y_value, color='red')

plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

plt.show()
```
## Output:
<img width="619" height="459" alt="image" src="https://github.com/user-attachments/assets/e02d8b4a-ebfd-46fc-bfee-989cbe7a81c3" />

<img width="261" height="36" alt="image" src="https://github.com/user-attachments/assets/befee05f-9e72-4a35-be0f-540bbfa5de63" />

<img width="601" height="454" alt="image" src="https://github.com/user-attachments/assets/bb57a617-e8e6-49b1-9261-82cd3435d785" />

<img width="1253" height="631" alt="image" src="https://github.com/user-attachments/assets/46d90038-e4b4-4a54-8da2-c8b5e4da2961" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
