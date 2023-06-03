

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# Stack Regerssor Over Sampling
#  'ridge__alpha', 'lasso__alpha', 'svr__C', 'svr__kernel
# Load the dataset
df = pd.read_csv("LungCancer24.csv")

# Define the data
df = df[['Age','Race','Sex','Site specific surgery','Behavior','Primary Site','Laterality',
      'Total tumors','TNM AJCC','Reason no surgey','Histology',
      'Survival years']]

# Split data into features and target
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the base models
base_models = [
    ('linear', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('svr', SVR())
]

# Define the stacking regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Define the parameter grid for grid search
param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0],
    'lasso__alpha': [0.1, 1.0, 10.0],
    'svr__C': [0.1, 1.0, 10.0],
    'svr__kernel': ['linear', 'rbf']
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(stacking_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Fit the best model on the training data
best_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Calculate regression metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
std_dev = y_test.std()
std_residuals = (y_test - y_pred).std()
y_mean = y_test.mean()
w = std_dev / std_residuals

# Print the regression metrics and weighting factor
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
print(f'Standard deviation: {std_dev}')
print(f'Standard deviation of residuals: {std_residuals}')
print(f'Mean of the target variable: {y_mean}')
print(f'Weighting Factor: {w}')


# Plot predicted vs actual survival times
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Survival Time (years)')
plt.ylabel('Predicted Survival Time (years)')
plt.title('Stacking Regressor Over-Sampling')
plt.show()
