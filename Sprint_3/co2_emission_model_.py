import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv('c:/Users/user/Desktop/project/CO2 Emissions_Canada.csv')  # Replace with your actual dataset path

# Preprocessing steps
# 1. Remove duplicates
duplicate = data.duplicated().sum()
print('There are {} duplicated rows in the data'.format(duplicate))
data.drop_duplicates(inplace=True)
data.reset_index(inplace=True, drop=True)

# 2. Detect and remove outliers using the IQR method
df_num_features = data.select_dtypes(include=np.number)
Q1 = df_num_features.quantile(0.25)
Q3 = df_num_features.quantile(0.75)
IQR = Q3 - Q1
print('IQR values for numerical columns:')
print(IQR)

# Condition to detect outliers
outlier_condition = (df_num_features < (Q1 - 1.5 * IQR)) | (df_num_features > (Q3 + 1.5 * IQR))

# Print number of outliers per column
for i in outlier_condition.columns:
    print('Total number of outliers in column {} are {}'.format(i, outlier_condition[i].sum()))

# Remove rows with any outliers
non_outlier_mask = ~outlier_condition.any(axis=1)
data_no_outliers = data[non_outlier_mask]
data_no_outliers.reset_index(inplace=True, drop=True)
data = data_no_outliers  # Updating the original data with no outliers

# Prepare features and target variable
X = data.drop(['CO2 Emissions(g/km)'], axis=1)
y = data['CO2 Emissions(g/km)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Specify categorical features
cat_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

# Initialize and train the CatBoost model
model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.01,
    depth=10,
    loss_function='RMSE',
    l2_leaf_reg=5,
    cat_features=cat_features,
    verbose=500
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

# Predictions for the training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate error metrics for the training set
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

# Calculate error metrics for the test set
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

# Print metrics
print(f"Training Data Metrics:")
print(f"MAE: {mae_train}")
print(f"RMSE: {rmse_train}")
print(f"R²: {r2_train}")

print(f"\nTest Data Metrics:")
print(f"MAE: {mae_test}")
print(f"RMSE: {rmse_test}")
print(f"R²: {r2_test}")

# Save the model using pickle
model_filename = 'co2_emission_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f'Model has been saved to {model_filename}')
