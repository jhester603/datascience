# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:15:32 2022

@author: JHest
"""

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in dataframe
df = pd.read_csv('titanic_train.csv')

"""
# =============================================
# Exploratory Analysis
# =============================================
"""
# First 50 rows
first_50 = df.head(50)

# Datatypes
df.dtypes

# Info 
df.info()

# Basic descriptive statistics
descr = df.describe()

# Value counts
num_vars = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived', 'Pclass']
cat_vars = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'] # Pclass and Survived could be treated as categorical as well

df['Sex'].value_counts()
df['Survived'].value_counts()
df['Embarked'].value_counts()
df['Pclass'].value_counts()



# =============================================
# Exploratory Analysis: Visualizations
# =============================================


# Histogram
df.hist(bins=50, figsize=(20,15))
plt.show()


# Distributions

def show_distribution(var_data):
	# Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]
    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\n Mode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val, mean_val, med_val, mod_val, max_val)) 
    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))
    
    # Plot the histogram   
    ax[0].hist(var_data, color='lightblue', ec='black', density=True)
    ax[0].set_ylabel('Frequency')

	# Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

	# Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

	# Add a title to the Figure
    fig.suptitle('Data Distribution')

	# Show the figure
    fig.show()


col = df['Age']
show_distribution(col)

# ============================
# Box plots
# ============================

df.boxplot('Age', figsize=(8,5))
df.boxplot('Fare')  # Note the outlier with fare above 500

df.boxplot('Parch')
df.boxplot('SibSp') # Note outlier at 8




"""
# =============================================
# Data Cleansing
# =============================================
"""

# =============================================
# Outliers
# =============================================

## Removing Outliers (Using 3 standard deviations)

def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-3*IQR)) | (df>(q3+3*IQR)))] 
   return(outliers)

outliers = find_outliers_IQR(df['Fare'])


print('number of outliers: '+ str(len(outliers)))
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min()))
outliers


df = df[df['Fare'] < 107]

outliers = find_outliers_IQR(df['SibSp'])

print('number of outliers: '+ str(len(outliers)))
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min()))
outliers

df = df[df['SibSp'] < 5]


# =============================================
# Handling Na values
# =============================================

print(df.isna().sum())

# The majority of Cabin values are na, let's drop this column
df = df.drop(columns=['Cabin'])

# Embarked: 2 missing values
df['Embarked'].value_counts() # Fill with most likely value
df['Embarked'] = df['Embarked'].fillna('S')


# Age: 177 missing values
177 / df.Survived.count() # About 20% missing age values


# =====================================
# Using ML to fill in age values
# =====================================

# Create 1hot encoder for sex
gender = []
for i in df['Sex']:
    if i == 'male':
        gender.append(1)
    else:
        gender.append(0)

df['Gender'] = gender

# # One-hot encoding Embarked
embarked = pd.DataFrame(df['Embarked'])

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
embarked_1hot = cat_encoder.fit_transform(embarked)

# Turn into array
embarked_encoded = embarked_1hot.toarray()
embarked_encoded = pd.DataFrame(embarked_encoded)

# Get categories in embarked_encoded
cat_encoder.categories_

# Add to dataframe
df['Embarked (C)'] = embarked_encoded[0]
df['Embarked (Q)'] = embarked_encoded[1]
df['Embarked (S)'] = embarked_encoded[2]


# Create dataframe containing missing age values
na_age = df[df['Age'].isna()]

# Create copy of dataframe
df_copy = df.copy()

# Drop nas 
df_copy.dropna(how='any', inplace=True)



# =============================
# Modeling
# =============================

# Split label from explanatories
y = df_copy['Age']
X = df_copy.drop(columns=['Age', 'Name', 'Sex', 'Ticket']) #Filter out cat_vars


# Split train/test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_copy, test_size=0.1, random_state=42)

# Trains
X_train = train_set.drop(columns=['Age', 'Name', 'Sex', 'Ticket', 'Embarked'])
y_train = train_set['Age']

# Tests
X_test = test_set.drop(columns=['Age', 'Name', 'Sex', 'Ticket', 'Embarked'])
y_test = test_set['Age']


# Full
y = df_copy['Age']
X = df_copy.drop(columns=['Age', 'Name', 'Sex', 'Ticket', 'Embarked']) #Filter out cat_vars



from sklearn.ensemble import RandomForestRegressor 
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)

rf_preds = forest_reg.predict(X_test)



# Performance Metrics
import sklearn.metrics as metrics
def regression_results(y_test, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_test, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
    mse=metrics.mean_squared_error(y_test, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
    r2=metrics.r2_score(y_test, y_pred)
    errors = abs(y_pred - y_test)
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    
   
    print('Accuracy:', round(accuracy, 2), '%.')
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

regression_results(y_test, rf_preds)
abs_errors = abs(rf_preds - y_test)

# Better than filling in with mean or median

X_na = na_age.drop(columns = ['Age', 'Name', 'Sex', 'Ticket', 'Embarked'])


# Filling in Missing age values
from sklearn.ensemble import RandomForestRegressor 
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X, y)

age_preds = forest_reg.predict(X_na)

na_age['Age'] = age_preds

# Add back to df_copy
df_full = pd.concat([df_copy,na_age], ignore_index=True)


"""
# =============================================
# Modeling Using Regression
# =============================================
"""


df_full = df_full.drop(columns=['Name', 'Sex', 'Ticket', 'Embarked'])

# Modeling
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_full, test_size=0.1, random_state=42)

X_train = train_set.drop(columns=['Survived'])
y_train = train_set['Survived']

X_test = test_set.drop(columns='Survived')
y_test = test_set['Survived']


from sklearn.ensemble import RandomForestRegressor 
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)

# Predictions
rf_preds = forest_reg.predict(X_test)

# Results
regression_results(y_test, rf_preds)

# Probability >= 0.5 is classified as 1 (Survived)
binary_preds = []
for i in df['Survived']:
    if i < 0.5:
        i = 0
    else:
        i = 1
    binary_preds.append(i)

results = y_test - binary_preds

results.value_counts()

rf_error_rate = 21 / 72





"""
# =============================================
# Binary Classification
# =============================================
"""
# =============================================
# SGD
# =============================================

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)

sgd_preds = sgd.predict(X_test)

sgd_results = y_test - sgd_preds
sgd_results.value_counts()

sgd_error_rate = 29/ 72


# =============================================
# RandomForestClassification
# =============================================

from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier(random_state=42)
forest_class.fit(X_train, y_train)

forest_class_preds = forest_class.predict(X_test)
forest_class_results = y_test - forest_class_preds
forest_class_results.value_counts()

forest_class_ = 10/83


# =============================================
# XGBoost Classification
# =============================================

# Metrics
def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_cur(fper, tper)
    
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    
    return model, accuracy, roc_auc, coh_kap, time_taken



# Training
import xgboost as xgb
params_xgb ={'n_estimators': 300,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)




"""
# =============================================
# Test Set
# =============================================
"""

# Read in test dataframe
test = pd.read_csv('titanic_test.csv')

# nas
print(test.isna().sum())

# One missing fare value
fare_na = test[test['Fare'].isna()]
male = test[test['Sex'] == 'male']
male_class3 = male[male['Pclass'] == 3]
male_class3_60 = male_class3[male_class3['Age'] > 30]
missing_fare = male_class3_60['Fare'].median()

test['Fare'].fillna(missing_fare, inplace=True)


test = test.drop(columns=['Cabin'])

# Embarked: 2 missing values
test['Embarked'].value_counts() # Fill with most likely value
test['Embarked'] = test['Embarked'].fillna('S')



# ===============================
# Using ML to fill in age values
# ===============================

# Create 1hot encoder for sex
gender = []
for i in test['Sex']:
    if i == 'male':
        gender.append(1)
    else:
        gender.append(0)

test['Gender'] = gender

# Encode Embarked
embarked = pd.DataFrame(test['Embarked'])

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
embarked_1hot = cat_encoder.fit_transform(embarked)

# Turn into array
embarked_encoded = embarked_1hot.toarray()
embarked_encoded = pd.DataFrame(embarked_encoded)

# Get categories in embarked_encoded
cat_encoder.categories_

# Add to dataframe
test['Embarked (C)'] = embarked_encoded[0]
test['Embarked (Q)'] = embarked_encoded[1]
test['Embarked (S)'] = embarked_encoded[2]




# Create dataframe containing missing age values
na_age = test[test['Age'].isna()]

# Create copy of dataframe
test_copy = test.copy()

# Drop nas
test_copy.dropna(how='any', inplace=True)



# Split label from explanatories
y = test_copy['Age']
X = test_copy.drop(columns=['Age', 'Name', 'Sex', 'Ticket']) #Filter out cat_vars


# Full
y = test_copy['Age']
X = test_copy.drop(columns=['Age', 'Name', 'Sex', 'Ticket', 'Embarked']) #Filter out cat_vars



X_na = na_age.drop(columns = ['Age', 'Name', 'Sex', 'Ticket', 'Embarked'])



from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X, y)

age_preds = forest_reg.predict(X_na)


na_age['Age'] = age_preds

# Add back to test_copy
test_full = pd.concat([test_copy,na_age], ignore_index=True)


"""
FULL MODEL
"""

# Train and test sets
X_train = df_full[['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'Embarked (C)', 'Embarked (Q)', 'Embarked (S)', 'Age']] 
y_train = df_full['Survived']

X_test = test_full[['PassengerId','Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'Embarked (C)', 'Embarked (Q)', 'Embarked (S)', 'Age']] 
X_test.sort_values(by=['PassengerId'], inplace=True)

# Modeling

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 500,
             'random_state': 12345}




forest_class = RandomForestClassifier(**params_rf)
forest_class.fit(X_train, y_train)

forest_class_preds = forest_class.predict(X_test)

final_predictions = pd.DataFrame(forest_class_preds)
final_predictions['PassengerId'] = test['PassengerId']

submission = final_predictions[['PassengerId', 0]]
submission.columns=['PassengerId', 'Survived']

# Export
submission.to_csv('submission.csv', index=False)









