#import packages and libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor  # Import XGBoost
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Load the training and test data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test_9K3DBWQ_2aRGUxy.csv")

# impute missing values with mean
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

#onehotencoding categorical features
df_train = pd.get_dummies(df_train, columns=['account_type', 'gender', 'loan_enq'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['account_type', 'gender', 'loan_enq'], drop_first=True)

# Define features (X) and target variable (y)
X = df_train.drop(columns=['id', 'cc_cons'])
y = df_train['cc_cons']

q1 = df_train['cc_cons'].quantile(0.25)
q3 = df_train['cc_cons'].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr

a = df_train[df_train['cc_cons'] > upper].index
df_train.drop(a, inplace=True)

y = np.log1p(y)  
# Log transformation with a small constant to handle zeros


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train multiple regressor models
base_models = [
    LinearRegression(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    DecisionTreeRegressor(),
    XGBRegressor()
]

# Create an array to store predictions from base models
base_model_predictions = np.zeros((len(X_valid), len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_valid)
    y_pred = np.abs(y_pred)
    
    base_model_predictions[:, i] = y_pred

# Stack the predictions of base models using GradientBoostingRegressor
stacked_model = GradientBoostingRegressor()
stacked_model.fit(base_model_predictions, y_valid)

# Make predictions on the validation set using the stacked model
stacked_predictions = stacked_model.predict(base_model_predictions)
stacked_predictions = np.abs(stacked_predictions)

# Calculate RMSLE (Root Mean Squared Logarithmic Error) for the stacked model
rmsle_stacked = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(stacked_predictions)))  # Convert back from log
print(f'Stacked Model RMSLE: {rmsle_stacked}')

# Train the stacked model on the entire training dataset
stacked_model.fit(X, y)

 

# Prepare the test data (same as the training data)
X_test = df_test.drop(columns=['id'])

 

# Make predictions on the test data
test_predictions = model.predict(X_test)

 

# Inverse transform the predictions to get the original scale
test_predictions = np.expm1(test_predictions)

 

# Create a submission dataframe
submission_df = pd.DataFrame({'id': df_test['id'], 'cc_cons': test_predictions})

 

# Save the submission dataframe to a CSV file
submission_df.to_csv('finalsubmission.csv', index=False)



