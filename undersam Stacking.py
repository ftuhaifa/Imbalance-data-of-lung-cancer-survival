# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:45:32 2023

@author: ftuha
"""

#under Sampling Stacking 


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:58:37 2023
@author: ftuha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from mlxtend.classifier import StackingClassifier
from imblearn.under_sampling import RandomUnderSampler


# Load the dataset
df = pd.read_csv("LungCancer24.csv")

# Define the data
df = df[['Age','Race','Sex','Site specific surgery','Behavior','Primary Site','Laterality',
      'Total tumors','TNM AJCC','Reason no surgey','Histology',
      'Survival years']]

# Split data into features and target
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Behavior', 'Primary Site', 'Laterality', 'Site specific surgery', 'Total tumors', 'Race', 'TNM AJCC', 'Reason no surgey', 'Histology'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Perform random undersampling
rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
rfc = RandomForestClassifier(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)
mlp = MLPClassifier(random_state=42)

# Define stacking classifier
sclf = StackingClassifier(
    classifiers=[rfc, gbc, mlp],
    meta_classifier=RandomForestClassifier(random_state=42),
    use_probas=True,
    average_probas=False
)

# Define parameter grid for grid search
params = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'gradientboostingclassifier__n_estimators': [50, 100, 200],
    'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (200,)]
}

# Perform grid search
grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
grid.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters: ", grid.best_params_)
print("Best Score: ", grid.best_score_)

#************************************************
# Compute ROC AUC

y_score = grid.predict_proba(X_test)

# Generate predicted probabilities using your classifier
# Ensure y_score is 2-dimensional
if y_score.ndim == 1:
    y_score = np.vstack([1 - y_score, y_score]).T

# Check the shape of y_score
print(y_score.shape)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')

#**********************************************************

# Predict on the test data
y_pred = grid.predict(X_test)
# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print("Accuracy: {:.2f}".format(accuracy))
print("F1-score: {:.2f}".format(f1))
print("Average AUC: {:.2f}".format(roc_auc))
print("\nClassification Report:")



# Use best estimator





import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc

y_prob = grid.best_estimator_.predict_proba(X_test)
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