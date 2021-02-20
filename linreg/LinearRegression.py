import array

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.ticklabel_format(style='plain')


class LinearRegression:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)

        self.x_name = self.data.columns[0]
        self.y_name = self.data.columns[1]

        self.x = self.data[self.x_name]
        self.y = self.data[self.y_name]

        self.b1 = 0
        self.b0 = 0

        self.loss_over_iterations = []

    # Plots the data, including the linear regression line, if the model has been trained.
    def plot_model_data(self):
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.scatter(self.data[self.x_name], self.data[self.y_name], c="black")

        if self.b1 != 0 and self.b0 != 0:
            pred_y = self.b1 * self.x + self.b0
            plt.plot([min(self.x), max(self.x)], [min(pred_y), max(pred_y)], color='blue')  # regression line
        plt.show()

    def plot_loss(self):
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.scatter(range(300), self.loss_over_iterations)
        plt.show()


    def train(self, learning_rate=0.0001, iterations=500):
        n = float(len(self.x))  # Number of samples

        self.b1 = 0  # Resets the values, if one wants to train the model again. The reason for wanting
        self.b0 = 0  # to train a model again can be to change learning rate, amount of iterations, etc.

        # Trains the model
        for i in range(iterations):
            predicted_y = self.b1 * self.x + self.b0

            der_b1 = (-2 / n) * sum(self.x * (self.y - predicted_y))  # Partial derivative of b1.
            der_b0 = (-2 / n) * sum(self.y - predicted_y)  # Partial derivative of b0.

            self.b1 = self.b1 - learning_rate * der_b1
            self.b0 = self.b0 - learning_rate * der_b0

        print(self.b1, self.b0)

    def predict(self, x):
        if self.b1 != 0 and self.b0 != 0:
            predicted_y = self.b1 * x + self.b0
            print("The predicted value is: " + str(predicted_y))
        else:
            print("Please train the model before attempting to make any predictions!")
