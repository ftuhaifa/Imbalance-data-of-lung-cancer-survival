# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:00:47 2023

@author: ftuha
"""



#Fuzzy Grid Boost Tree without fuzzy random forest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import skfuzzy as fuzz

# Load and preprocess the data
df = pd.read_csv("LungCancer25.csv")

# Define the data
df = df[['Age','Race','Sex','Site specific surgery','Behavior','Primary Site',
         'Laterality',
         'Total tumors','TNM AJCC','Reason no surgey','Histology',
         'Survival years']]




#Fuzzify the input data: Convert the crisp input data into fuzzy sets
#using appropriate fuzzy membership functions. You can use the skfuzzy
#package in Python for this purpose.

fuzzy_sets = []

# Define the universe of discourse for Age with a step of 5
x_age = np.arange(0, 121, 5)

# Define the fuzzy membership functions for the Age categories
child = fuzz.trimf(x_age, [0, 7.5, 15])
young_adult = fuzz.trimf(x_age, [20, 25, 30])
middle_aged_mf = fuzz.trimf(x_age, [30, 40, 59])
old_age = fuzz.trapmf(x_age, [60, 65, 120, 120])

# Fuzzify the input data (Age column)
fuzzy_age_data = []

for age in df['Age']:
    child_degree = fuzz.interp_membership(x_age, child, age)
    young_adult_degree = fuzz.interp_membership(x_age, young_adult, age)
    middle_aged_degree = fuzz.interp_membership(x_age, middle_aged_mf, age)
    old_age_degree = fuzz.interp_membership(x_age, old_age, age)
    
    fuzzy_age_data.append([child_degree,
                           young_adult_degree,
                           middle_aged_degree,
                           old_age_degree])

# Add fuzzified age data to DataFrame
df_fuzzy_age = pd.DataFrame(fuzzy_age_data, columns=['Child',
                                                     'Young_Adult',
                                                     'Middle_Aged',
                                                     'Old_Age'])
#df_fuzzy = pd.concat([df, df_fuzzy_age], axis=1)

#print(df_fuzzy)

print("Child degrees:", child_degree)
print("Young adult degrees:", young_adult_degree)
print("Middle-aged degrees:", middle_aged_degree)
print("Old age degrees:", old_age_degree)



#*****************************************************************************
#*****************************************************************************

import skfuzzy as fuzz

# Define the universe of discourse for Race
x_race = np.arange(0, 6, 1)

# Define the fuzzy membership functions for the Race categories
race_1 = fuzz.trapmf(x_race, [1, 1, 1.5, 2.5])
race_2 = fuzz.trapmf(x_race, [2, 2, 2.5, 3.5])
race_3 = fuzz.trapmf(x_race, [3, 3, 3.5, 4.5])
race_4 = fuzz.trapmf(x_race, [4, 4, 4.5, 5.5])
race_5 = fuzz.trapmf(x_race, [5, 5, 5.5, 5.5])

# Fuzzify the input data (Race column)
fuzzy_race_data = []

for race in df['Race']:
    race_1_degree = fuzz.interp_membership(x_race, race_1, race)
    race_2_degree = fuzz.interp_membership(x_race, race_2, race)
    race_3_degree = fuzz.interp_membership(x_race, race_3, race)
    race_4_degree = fuzz.interp_membership(x_race, race_4, race)
    race_5_degree = fuzz.interp_membership(x_race, race_5, race)

    fuzzy_race_data.append([race_1_degree,
                            race_2_degree,
                            race_3_degree,
                            race_4_degree,
                            race_5_degree])
    
print("race_1_degree:", race_1_degree)
print("race_2_degree:", race_2_degree)
print("race_3_degree:", race_3_degree)
print("race_4_degree:", race_4_degree)
print("race_5_degree:", race_5_degree)


#*****************************************************************************
#*****************************************************************************



# Define the universe of discourse for Sex
x_sex = np.arange(1, 3, 1)

# Define the fuzzy membership functions for the Sex categories
male = fuzz.trimf(np.array(x_sex), [1, 1, 2])
female = fuzz.trimf(np.array(x_sex), [1, 2, 2])

# Fuzzify the input data (Sex column)
fuzzy_sex_data = []

for sex in df['Sex']:
    male_degree = fuzz.interp_membership(x_sex, male, sex)
    female_degree = fuzz.interp_membership(x_sex, female, sex)
    fuzzy_sex_data.append([male_degree, female_degree])
    
fuzzy_sex_data = np.array(fuzzy_sex_data)


print("male_degree:", male_degree)
print("female_degree:", female_degree)



#*****************************************************************************
#*****************************************************************************



# Define the universe of discourse for Site specific surgery
x_surgery = np.arange(0, 81, 1)

# Define the fuzzy membership functions for the Site specific surgery categories
no_surgery = fuzz.trimf(x_surgery, [0, 0, 10])
minor_surgery = fuzz.trimf(x_surgery, [9, 20, 30])
moderate_surgery = fuzz.trimf(x_surgery, [20, 40, 50])
major_surgery = fuzz.trimf(x_surgery, [50, 70, 80])

# Fuzzify the input data (Site specific surgery column)
fuzzy_surgery_data = []

for surgery in df['Site specific surgery']:
    no_surgery_degree = fuzz.interp_membership(x_surgery, no_surgery, surgery)
    minor_surgery_degree = fuzz.interp_membership(x_surgery,
                                                  minor_surgery, surgery)
    moderate_surgery_degree = fuzz.interp_membership(x_surgery,
                                                     moderate_surgery, surgery)
    major_surgery_degree = fuzz.interp_membership(x_surgery,
                                                  major_surgery, surgery)

    fuzzy_surgery_data.append([no_surgery_degree,
                               minor_surgery_degree,
                               moderate_surgery_degree,
                               major_surgery_degree])




print("no_surgery_degree:", no_surgery_degree)
print("minor_surgery_degree:", minor_surgery_degree)
print("moderate_surgery_degree:", moderate_surgery_degree)
print("major_surgery_degree:", major_surgery_degree)





#*****************************************************************************
#*****************************************************************************



# Define the universe of discourse for Behavior
x_behavior = np.arange(0, 5, 1)



# Define the fuzzy membership functions for the Behavior categories
in_situ = fuzz.trimf(x_behavior, [1, 1, 2])
malignant = fuzz.trimf(x_behavior, [2, 3, 3])

# Fuzzify the input data (Behavior column)
fuzzy_behavior_data = []

for behavior in df['Behavior']:
    in_situ_degree = fuzz.interp_membership(x_behavior, in_situ, behavior)
    malignant_degree = fuzz.interp_membership(x_behavior, malignant, behavior)

    fuzzy_behavior_data.append({
        'in_situ': in_situ_degree,
        'malignant': malignant_degree,
    })

print("in_situ_degree:", in_situ_degree)
print("malignant_degree:", malignant_degree)




#*****************************************************************************
#*****************************************************************************

# Define the universe of discourse for Primary Site
x_site = np.arange(340, 350, 1)



# Define the fuzzy membership functions for the Primary Site categories
site_340 = fuzz.trimf(x_site, [340, 340, 341])
site_341 = fuzz.trimf(x_site, [340, 341, 342])
site_342 = fuzz.trimf(x_site, [341, 342, 343])
site_343 = fuzz.trimf(x_site, [342, 343, 344])
site_348 = fuzz.trimf(x_site, [343, 348, 349])
site_349 = fuzz.trimf(x_site, [348, 349, 349])



# Fuzzify the input data (Primary Site column)
fuzzy_site_data = []

for site in df['Primary Site']:
    site_340_degree = fuzz.interp_membership(x_site, site_340, site)
    site_341_degree = fuzz.interp_membership(x_site, site_341, site)
    site_342_degree = fuzz.interp_membership(x_site, site_342, site)
    site_343_degree = fuzz.interp_membership(x_site, site_343, site)
    site_348_degree = fuzz.interp_membership(x_site, site_348, site)
    site_349_degree = fuzz.interp_membership(x_site, site_349, site)
    
    fuzzy_site_data.append((site_340_degree, site_341_degree, site_342_degree, site_343_degree, site_348_degree, site_349_degree))


print("site_340_degree:", site_340_degree)
print("site_341_degree:", site_341_degree)
print("site_342_degree:", site_342_degree)
print("site_343_degree:", site_343_degree)
print("site_348_degree:", site_342_degree)
print("site_349_degree:", site_343_degree)

#*****************************************************************************
#*****************************************************************************



# Define the universe of discourse for Laterality
x_laterality = np.arange(0, 10, 1)

 

# Define the fuzzy membership functions for the Laterality categories
not_paired = fuzz.trimf(x_laterality, [0, 0, 1])
right_origin = fuzz.trimf(x_laterality, [1, 1, 2])
left_origin = fuzz.trimf(x_laterality, [2, 2, 3])
one_side = fuzz.trimf(x_laterality, [3, 3, 4])
bilateral = fuzz.trimf(x_laterality, [4, 4, 5])
midline_tumor = fuzz.trimf(x_laterality, [5, 5, 6])
paired_site = fuzz.trimf(x_laterality, [6, 6, 7])
unknown = fuzz.trimf(x_laterality, [7, 7, 8])
no_info = fuzz.trimf(x_laterality, [8, 8, 9])




# Fuzzify the input data (Laterality column)
fuzzy_laterality_data = []

for laterality in df['Laterality']:
    not_paired_degree = fuzz.interp_membership(x_laterality, not_paired, laterality)
    right_origin_degree = fuzz.interp_membership(x_laterality, right_origin, laterality)
    left_origin_degree = fuzz.interp_membership(x_laterality, left_origin, laterality)
    one_side_degree = fuzz.interp_membership(x_laterality, one_side, laterality)
    bilateral_degree = fuzz.interp_membership(x_laterality, bilateral, laterality)
    midline_tumor_degree = fuzz.interp_membership(x_laterality, midline_tumor, laterality)
    paired_site_degree = fuzz.interp_membership(x_laterality, paired_site, laterality)
    unknown_degree = fuzz.interp_membership(x_laterality, unknown, laterality)
    no_info_degree = fuzz.interp_membership(x_laterality, no_info, laterality)
    
    fuzzy_laterality_data.append([not_paired_degree, right_origin_degree,
                                  left_origin_degree, one_side_degree,
                                  bilateral_degree, midline_tumor_degree, paired_site_degree, unknown_degree,
                                  no_info_degree])
    
# Print the degree for each Laterality category
for i, laterality in enumerate(['Not a paired site', 'Right: origin of primary', 'Left: origin of primary',
                                'Only one side involved', 'Bilateral involvement', 'Paired site: midline tumor',
                                'Paired site', 'Unknown', 'Paired site but no information']):
    print(laterality, ": ", fuzzy_laterality_data[0][i])
    
#*****************************************************************************
#*****************************************************************************


# Define the universe of discourse for Total tumors
x_total_tumors = np.arange(0, 6, 1)



# Define the fuzzy membership functions for the Total tumors categories
few_tumors = fuzz.trimf(x_total_tumors, [0, 0, 2])
some_tumors = fuzz.trimf(x_total_tumors, [1, 2, 3])
many_tumors = fuzz.trimf(x_total_tumors, [2, 4, 5])




# Fuzzify the input data (Total tumors column)
fuzzy_total_tumors_data = []

for tumors in df['Total tumors']:
    few_tumors_degree = fuzz.interp_membership(x_total_tumors, few_tumors, tumors)
    some_tumors_degree = fuzz.interp_membership(x_total_tumors, some_tumors, tumors)
    many_tumors_degree = fuzz.interp_membership(x_total_tumors, many_tumors, tumors)
    
    fuzzy_total_tumors_data.append({'few_tumors': few_tumors_degree,
                                    'some_tumors': some_tumors_degree,
                                    'many_tumors': many_tumors_degree})
    
# Print the degree of membership for each fuzzy set
# Print the degree for each Total tumors category
print("few_tumors_degree:", few_tumors_degree)
print("some_tumors_degree:", some_tumors_degree)
print("many_tumors_degree:", many_tumors_degree)

#*****************************************************************************
#*****************************************************************************

# Define the universe of discourse for TNM AJCC
x_tnm = np.arange(0, 41, 1)



# Define the fuzzy membership functions for the TNM AJCC categories
occult = fuzz.trimf(x_tnm, [39, 39, 40])
unknown = fuzz.trimf(x_tnm, [39, 40, 40])
zero = fuzz.trimf(x_tnm, [0, 0, 1])
ia = fuzz.trimf(x_tnm, [10, 11, 12])
ib = fuzz.trimf(x_tnm, [20, 21, 22])
iia = fuzz.trimf(x_tnm, [11, 12, 13])
iib = fuzz.trimf(x_tnm, [21, 22, 23])
iiia = fuzz.trimf(x_tnm, [12, 13, 14])
iiib = fuzz.trimf(x_tnm, [22, 23, 24])
iv = fuzz.trimf(x_tnm, [30, 31, 32])




# Fuzzify the input data (TNM AJCC column)
fuzzy_tnm_data = []

for tnm in df['TNM AJCC']:
    occult_degree = fuzz.interp_membership(x_tnm, occult, tnm)
    unknown_degree = fuzz.interp_membership(x_tnm, unknown, tnm)
    zero_degree = fuzz.interp_membership(x_tnm, zero, tnm)
    ia_degree = fuzz.interp_membership(x_tnm, ia, tnm)
    ib_degree = fuzz.interp_membership(x_tnm, ib, tnm)
    iia_degree = fuzz.interp_membership(x_tnm, iia, tnm)
    iib_degree = fuzz.interp_membership(x_tnm, iib, tnm)
    iiia_degree = fuzz.interp_membership(x_tnm, iiia, tnm)
    iiib_degree = fuzz.interp_membership(x_tnm, iiib, tnm)
    iv_degree = fuzz.interp_membership(x_tnm, iv, tnm)
    
    fuzzy_tnm_data.append([occult_degree, unknown_degree, zero_degree, ia_degree, ib_degree, iia_degree, 
                           iib_degree, iiia_degree, iiib_degree, iv_degree])




print("Occult carcinoma: ", occult_degree)
print("Unknown stage: ", unknown_degree)
print("Stage 0: ", zero_degree)
print("Stage IA: ", ia_degree)
print("Stage IB: ", ib_degree)
print("Stage IIA: ", iia_degree)
print("Stage IIB: ", iib_degree)
print("Stage IIIA: ", iiia_degree)
print("Stage IIIB: ", iiib_degree)
print("Stage IV: ", iv_degree)

#*****************************************************************************
#*****************************************************************************

# Define the universe of discourse for Reason no surgery
x_reason = np.arange(0, 4, 1)



# Define the fuzzy membership functions for the Reason no surgery categories
performed = fuzz.trimf(x_reason, [0, 0, 0.5])
not_recommended = fuzz.trimf(x_reason, [0.5, 1, 1.5])
not_performed = fuzz.trimf(x_reason, [1.5, 2, 2.5])
unknown_if_performed = fuzz.trimf(x_reason, [2.5, 3, 3])




# Fuzzify the input data (Reason no surgery column)
fuzzy_reason_data = []

for reason in df['Reason no surgey']:
    performed_degree = fuzz.interp_membership(x_reason, performed, reason)
    not_recommended_degree = fuzz.interp_membership(x_reason, not_recommended, reason)
    not_performed_degree = fuzz.interp_membership(x_reason, not_performed, reason)
    unknown_if_performed_degree = fuzz.interp_membership(x_reason, unknown_if_performed, reason)
    
    fuzzy_reason_data.append([performed_degree, not_recommended_degree, not_performed_degree, unknown_if_performed_degree])

# Print the degree of membership for each fuzzy set
print("Performed degree:", fuzzy_reason_data[0][0])
print("Not recommended degree:", fuzzy_reason_data[1][1])
print("Not performed degree:", fuzzy_reason_data[2][2])
print("Unknown if performed degree:", fuzzy_reason_data[3][3])

#*****************************************************************************
#*****************************************************************************


# Define the universe of discourse for Histology
x_histology = np.arange(1, 10)



# Define the fuzzy membership functions for the Histology categories
adenocarcinoma = fuzz.trimf(x_histology, [1, 1, 2])
squamous_cell_carcinoma = fuzz.trimf(x_histology, [2, 2, 3])
small_cell_carcinoma = fuzz.trimf(x_histology, [3, 3, 4])
large_cell_carcinoma = fuzz.trimf(x_histology, [4, 4, 5])
other_non_small_cell_carcinoma = fuzz.trimf(x_histology, [5, 5, 6])
unspecified_neoplasms = fuzz.trimf(x_histology, [6, 6, 7])
mix_of_carcinoma_and_sarcoma = fuzz.trimf(x_histology, [7, 7, 8])
sarcoma = fuzz.trimf(x_histology, [8, 8, 9])
transitional_cell_carcinoma = fuzz.trimf(x_histology, [9, 9, 10])




# Fuzzify the input data (Histology column)
fuzzy_histology_data = []

for histology in df['Histology']:
    adenocarcinoma_degree = fuzz.interp_membership(x_histology, adenocarcinoma, histology)
    squamous_cell_carcinoma_degree = fuzz.interp_membership(x_histology, squamous_cell_carcinoma, histology)
    small_cell_carcinoma_degree = fuzz.interp_membership(x_histology, small_cell_carcinoma, histology)
    large_cell_carcinoma_degree = fuzz.interp_membership(x_histology, large_cell_carcinoma, histology)
    other_non_small_cell_carcinoma_degree = fuzz.interp_membership(x_histology, other_non_small_cell_carcinoma, histology)
    unspecified_neoplasms_degree = fuzz.interp_membership(x_histology, unspecified_neoplasms, histology)
    mix_of_carcinoma_and_sarcoma_degree = fuzz.interp_membership(x_histology, mix_of_carcinoma_and_sarcoma, histology)
    sarcoma_degree = fuzz.interp_membership(x_histology, sarcoma, histology)
    transitional_cell_carcinoma_degree = fuzz.interp_membership(x_histology, transitional_cell_carcinoma, histology)

    fuzzy_histology_data.append([adenocarcinoma_degree, squamous_cell_carcinoma_degree,
                                 small_cell_carcinoma_degree, large_cell_carcinoma_degree,
                                 other_non_small_cell_carcinoma_degree, unspecified_neoplasms_degree,
                                 mix_of_carcinoma_and_sarcoma_degree, sarcoma_degree,
                                 transitional_cell_carcinoma_degree])

# Print the degree of membership for each fuzzy set
for i, histology in enumerate(['Adenocarcinoma', 'Squamous cell carcinoma', 'Small cell carcinoma',
                                'Large cell carcinoma', 'Other Non small cell carcinoma',
                                'Unspecified neoplasms', 'Mix of carcinoma and sarcoma', 'Sarcoma',
                                'Transitional cell carcinoma']):
    print(histology, ": ", fuzzy_histology_data[1][i])

#*****************************************************************************
#*****************************************************************************


#end of Fuzzify the input data

#*****************************************************************************
#*****************************************************************************



#*****************************************************************************
#*****************************************************************************

from skfuzzy import control as ctrl
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Store the fuzzy membership functions in fuzzy_sets
fuzzy_sets = [child, young_adult, middle_aged_mf,
              old_age, race_1, race_2, race_3,
              race_4, race_5, male, female,
              no_surgery, minor_surgery, moderate_surgery,
              major_surgery, in_situ, malignant, 
              site_340, site_341, site_342, site_343, site_348,
              site_349, 
              not_paired, right_origin, left_origin, one_side,
              bilateral,
              midline_tumor, paired_site,
              unknown, no_info,
              few_tumors,
              some_tumors, many_tumors, occult, unknown, zero, ia, ib, iia,
              iib, iiia, iiib, iv, performed, not_recommended, not_performed,
              unknown_if_performed, adenocarcinoma, squamous_cell_carcinoma,
              small_cell_carcinoma, large_cell_carcinoma, other_non_small_cell_carcinoma,
              unspecified_neoplasms, mix_of_carcinoma_and_sarcoma,
              sarcoma, transitional_cell_carcinoma]

# Access x[i] for a given fuzzy_set[i]
x = [x_age, x_age, x_age, x_age,
     x_race, x_race, x_race, x_race, x_race,
     x_sex, x_sex,
     x_surgery, x_surgery, x_surgery, x_surgery,
     x_behavior, x_behavior,
     x_site, x_site, x_site, x_site, x_site, x_site,
     x_laterality, x_laterality, x_laterality, x_laterality, x_laterality,
     x_laterality, x_laterality, x_laterality, x_laterality,
     x_total_tumors, x_total_tumors, x_total_tumors,
     x_tnm, x_tnm, x_tnm, x_tnm, x_tnm, x_tnm, x_tnm, x_tnm, x_tnm, x_tnm,
     x_reason, x_reason, x_reason, x_reason,
     x_histology, x_histology, x_histology, x_histology, x_histology, x_histology,
     x_histology, x_histology, x_histology
     ]


import numpy as np
import skfuzzy as fuzz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class FuzzyGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                               learning_rate=learning_rate,
                                               max_depth=max_depth)
        self.fuzzy_sets = None
        self.x = None
    
    def _fuzzify(self, X):
        X_fuzzy = []
        for i in range(X.shape[1]):
            X_fuzzy.append(fuzz.interp_membership(self.x[i], self.fuzzy_sets[i], X[:, i]))
        return np.asarray(X_fuzzy).T
    
    def fit(self, X, y):
        self.x = [np.arange(X[:, i].min(), X[:, i].max(), 0.01) for i in range(X.shape[1])]
        self.fuzzy_sets = [fuzz.trimf(self.x[i], [np.percentile(X[:, i], 25),
                                                  np.percentile(X[:, i], 50),
                                                  np.percentile(X[:, i], 75)]) for i in range(X.shape[1])]
        X_fuzzy = self._fuzzify(X)
        self.clf.fit(X_fuzzy, y)
    
    def predict(self, X):
        X_fuzzy = self._fuzzify(X)
        return self.clf.predict(X_fuzzy)
    
    def predict_proba(self, X):
        X_fuzzy = self._fuzzify(X)
        return self.clf.predict_proba(X_fuzzy)
    
    
# Split data into features and target
X = df.drop("Survival years", axis=1)
y = df["Survival years"]

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X , y = smote.fit_resample(X, y)



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2,
                                                    random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

scaler = MinMaxScaler()

pipeline = Pipeline([
    ('scaler', scaler),
    ('clf', FuzzyGradientBoostingClassifier())
])


from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__learning_rate': [0.1, 0.5, 1.0],
    'clf__max_depth': [1, 2, 3]
}



clf = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# print best hyperparameters and score
print("Best hyperparameters: ", clf.best_params_)
print("Best score: ", clf.best_score_)




# Predict on test set

y_pred_proba = clf.predict_proba(X_test)
# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

# Print the results

print("Accuracy: {:.2f}".format(accuracy))
print("F1-score: {:.2f}".format(f1))
print("Average AUC: {:.2f}".format(roc_auc))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




y_prob = clf.predict_proba(X_test)
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



