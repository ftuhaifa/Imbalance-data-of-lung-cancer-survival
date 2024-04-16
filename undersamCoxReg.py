import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lifelines import CoxPHFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load and preprocess the data
df = pd.read_csv("LungCancer24.csv")

# Select relevant columns
df = df[['Age', 'Race', 'Sex', 'Site specific surgery', 'Behavior','Primary Site', 'Laterality',
         'Total tumors', 'TNM AJCC', 'Reason no surgey', 'Histology', 'Event Indicator', 'Survival years']]

# Drop missing values and encode categorical variables (uncomment if needed)
# df = df.dropna()
# df = pd.get_dummies(df, ...)

# Underample the data
print("Performing undersampling...")
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(df.drop('Survival years', axis=1), df['Survival years'])

# Combine resampled features and target into a single DataFrame
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Remove or transform low variance features
# Add code here to remove or transform features with low variance if necessary

# Calculate VIF for your features
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

# Calculate VIF for resampled features
vif_df = calculate_vif(df_resampled.drop(['Survival years', 'Event Indicator'], axis=1))  # Exclude target column
print(vif_df)

# Remove or combine features with high VIF (collinearity) if necessary

# Fit the Cox regression model using CoxPHFitter
print("Fitting Cox regression model...")
cox_model = CoxPHFitter(penalizer=0.1)  # Adjust penalizer if necessary
cox_model.fit(df_resampled, duration_col='Survival years', event_col='Event Indicator')

# Print the summary of the model
print(cox_model.summary)

# Calculate and print the C-index
c_index = cox_model.concordance_index_
print("C-index:", c_index)

# Additional code for plotting and further analysis...








from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# Define the number of bins for calibration based on your specific range (0 to 16)
n_bins = 17  # You can adjust this based on your specific needs




import matplotlib.pyplot as plt

# Plot the survival curves for a subset of individuals
# Select the first 10 individuals for plotting (you can change this as needed)
individuals_to_plot = X_resampled.iloc[:17]  # Updated from X_smogn to X_resampled

# Calculate the survival function for these individuals
survival_functions = cox_model.predict_survival_function(individuals_to_plot)

import matplotlib.pyplot as plt


survival_functions.plot()
plt.title('Undersampling Cox Regression')  # Updated title to reflect undersampling
plt.xlabel('Time to Death (years)')
plt.ylabel('Survival Probability')
plt.xlim(0, 16)  # Set the x-axis range from 0 to 16
plt.show()

# For the second plot
n = 17  # number of individuals to plot
individuals_to_plot = df.iloc[:n, :]
cox_model.predict_survival_function(individuals_to_plot).plot()

plt.title('Undersampling Cox Regression')  # Updated title to reflect undersampling
plt.xlabel('Time to Death (years)')
plt.ylabel('Survival Probability')
plt.xlim(0, 16)  # Set the x-axis range from 0 to 16
plt.show()

