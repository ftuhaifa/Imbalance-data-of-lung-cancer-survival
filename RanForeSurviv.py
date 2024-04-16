import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_c_index
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the data
df = pd.read_csv("LungCancer24.csv")
df = df[['Age', 'Race', 'Sex', 'Site specific surgery', 'Behavior', 'Primary Site', 'Laterality',
         'Total tumors', 'TNM AJCC', 'Reason no surgey', 'Histology', 'Event Indicator', 'Survival years']]

# Create a CoxPHFitter object
cph = CoxPHFitter()

# Fit the Cox regression model
cph.fit(df, duration_col='Survival years', event_col='Event Indicator')

# Calculate the C-index
c_index = lifelines_c_index(df['Survival years'], -cph.predict_partial_hazard(df), df['Event Indicator'])

# Split the data into training and testing sets for calibration
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survival years', 'Event Indicator'], axis=1),
                                                    df[['Survival years', 'Event Indicator']],
                                                    test_size=0.2, random_state=42)

# Define the number of bins for calibration based on your specific range (0 to 16)
n_bins = 17  # You can adjust this based on your specific needs

# Calculate Calibration Plot
prob_true, prob_pred = calibration_curve(
    y_test['Event Indicator'],
    cph.predict_survival_function(X_test).values[0],
    n_bins=n_bins
)




# Print the evaluation metrics
print("C-Index (Concordance Index): {:.2f}".format(c_index))

n = 17  # number of individuals to plot
individuals_to_plot = df.iloc[:n, :]
cph.predict_survival_function(individuals_to_plot).plot()

plt.title('Standard Cox Regression')
plt.xlabel('Time to Death (years)')
plt.ylabel('Survival Probability')
plt.show()




