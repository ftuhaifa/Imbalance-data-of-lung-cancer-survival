# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from scipy import stats
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler 

# Load and preprocess the data
# Load and preprocess the data
df = pd.read_csv("LungCancer24.csv")
df = df[['Age', 'Race', 'Sex', 'Site specific surgery', 'Behavior', 'Primary Site', 'Laterality',
         'Total tumors', 'TNM AJCC', 'Reason no surgey', 'Histology', 'Survival years']]

# One-hot encode categorical variables
categorical_columns = ['Age', 'Race', 'Sex', 'Site specific surgery', 'Behavior', 'Primary Site', 'Laterality',
                       'Total tumors', 'TNM AJCC', 'Reason no surgey', 'Histology']
df = pd.get_dummies(df, columns=categorical_columns)

# Split data into features and target
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Rest of your code for model training and evaluation...



#for standard

# Define classifiers
# Define classifiers with hyperparameters
# Define and initialize classifiers
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(200,), random_state=42)

# Define and initialize stacking classifier
sclf = StackingClassifier(
    classifiers=[rfc, gbc, mlp],
    meta_classifier=RandomForestClassifier(random_state=42),
    use_probas=True,
    average_probas=False
)

# Rest of your code for model training and evaluation...




# One-hot encode categorical variables


#{'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50}

# Number of runs and an empty list to store results
num_runs = 2
results = []

for seed in range(num_runs):
    # Balance the classes using over-sampling with the current seed
    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    #under_sampler = RandomUnderSampler(random_state=seed)
    #X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    #X_resampled, y_resampled = X, y

    # Split the data with the current seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=seed)
    
    


    # Define and train the model with the specified hyperparameters
    sclf.fit(X_train, y_train)  # Use the existing instance of sclf


    # Evaluate the model
    y_pred = sclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    y_prob = sclf.predict_proba(X_test)
    auc = roc_auc_score(pd.get_dummies(y_test).values, y_prob, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Store the results for this run
    results.append({'seed': seed, 'accuracy': accuracy, 'f1': f1, 'auc': auc, 'precision': precision, 'recall': recall})

# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results)

print (results_df)

# Print average and standard deviation of the metrics with confidence intervals
confidence_level = 0.95
for metric in ['accuracy', 'f1', 'auc', 'precision', 'recall']:
    mean = results_df[metric].mean()
    std_dev = results_df[metric].std()
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, num_runs - 1) * (std_dev / np.sqrt(num_runs))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    print(f"Average {metric.capitalize()}: {mean:.4f} (Std Dev: {std_dev:.4f}, {confidence_level * 100:.0f}% CI: {lower_bound:.4f}, {upper_bound:.4f})")