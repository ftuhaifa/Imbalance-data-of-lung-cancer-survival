# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 08:39:52 2023

@author: ftuha
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 08:38:27 2023

@author: ftuha
"""

#Standard GBT SMOTE

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:24:06 2023
@author: ftuha
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load and preprocess the data
df = pd.read_csv("LungCancer24.csv")

# Define the data
df = df[['Age','Race','Sex','Site specific surgery','Behavior','Primary Site','Laterality',
         'Total tumors','TNM AJCC','Reason no surgey','Histology','Survival years']]

# Split data into features and target
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Balance the classes using over-sampling
#smote = SMOTE(random_state=42)
#X, y = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Replace missing values with the mean of the corresponding feature
#imputer = SimpleImputer()
#X_train = imputer.fit_transform(X_train)
#X_test = imputer.transform(X_test)

# Define the model with the hyperparameters to tune
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameters to tune
params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by grid search
print('Best hyperparameters:', grid_search.best_params_)

# Define the model with the best hyperparameters
best_model = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)

# Fit the model to the training data
best_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Compute the accuracy, F-1 score, and AUC
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
y_prob = best_model.predict_proba(X_test)
n_classes = len(np.unique(y))
auc = roc_auc_score(pd.get_dummies(y_test).values, y_prob, average='weighted')

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(pd.get_dummies(y_test).iloc[:, i], y_prob[:, i])
    roc_auc[i] = roc_auc_score(pd.get_dummies(y_test).iloc[:, i], y_prob[:, i])

# Plot ROC curve
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Print the accuracy, F-1 score, and AUC
print('Accuracy:', accuracy)
print('F-1 score:', f1)
print('Average AUC:', auc)
for i in range(n_classes):
    print(f'AUC for Class {i}:', roc_auc[i])
