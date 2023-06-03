# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:05:24 2023

@author: ftuha
"""

#MLPRegressor stands for Multi-Layer Perceptron Regressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv("LungCancer24.csv")

# Define the data
df = df[['Age','Race','Sex','Site specific surgery','Behavior','Primary Site','Laterality',
         'Total tumors','TNM AJCC','Reason no surgey','Histology','Survival years']]

# Split data into features and target
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

# Balance the classes using over-sampling
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Define the hyperparameters and their range of values
param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'learning_rate': ['constant', 'invscaling', 'adaptive'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
              'batch_size': [32, 64, 128]}

# Create a GridSearchCV object
grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Train a new model with the best hyperparameters
regressor = MLPRegressor(**grid_search.best_params_)
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
std_dev = np.std(y_test)
std_residuals = np.std(y_test - y_pred)
y_mean = np.mean(y_test)
w = std_dev / y_mean

# Print evaluation metrics
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
print(f'Standard deviation: {std_dev}')
print(f'standard deviation of residuals: {std_residuals}')
print(f'mean of the target variable: {y_mean}')
print(f'Weighting Factor: {w}')

# Plot predicted vs actual survival times
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Survival Time (years)')
plt.ylabel('Predicted Survival Time (years)')
plt.title('Neural network with over-sampling')
plt.show()





