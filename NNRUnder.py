# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:08:52 2023

@author: ftuha
"""
#MLPRegressor stands for Multi-Layer Perceptron Regressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv("LungCancer24.csv")
df = df[['Age','Race','Sex','Site specific surgery','Behavior','Primary Site','Laterality',
         'Total tumors','TNM AJCC','Reason no surgey','Histology','Survival years']]
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the under-sampling strategy
undersampler = RandomUnderSampler(random_state=42)

# Define the hyperparameters and their range of values
# Define the parameter grid to search
param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'learning_rate': ['constant', 'invscaling', 'adaptive'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
              'batch_size': [32, 64, 128]}

# Create a GridSearchCV object with under-sampling and fit it to the training data
grid_search = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")
# Extract the best model from the GridSearchCV object and predict on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute regression metrics on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
std_dev = np.std(y_test)
std_residuals = np.std(y_test - y_pred)
y_mean = np.mean(y_test)
w = np.sqrt(np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - y_mean)))

# Print the regression metrics and weighting factor
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
plt.title('Multi-Layer Perceptron Regressor under-sampling')
plt.show()
