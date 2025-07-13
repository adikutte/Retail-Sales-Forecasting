import pandas as pd

# Load the dataset from your folder
train_df = pd.read_csv(r"C:\Users\ADITYA\OneDrive\Desktop\New folder\train.csv")

# Convert 'date' column to datetime
train_df['date'] = pd.to_datetime(train_df['date'])

# Check results
print(train_df.dtypes)
print(train_df.head())

# Check for missing values
print(train_df.isnull().sum())
# Sorting data by date
train_df = train_df.sort_values("date")
# check first few rows again
print(train_df.head())


import matplotlib.pyplot as plt

# Group by date and sum sales
daily_sales = train_df.groupby("date")["sales"].sum()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(daily_sales)
plt.title("Total Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# Extract month from date
train_df['month'] = train_df['date'].dt.month

# Calculate average sales per month
monthly_avg_sales = train_df.groupby('month')['sales'].mean()

# Plot
plt.figure(figsize=(10, 5))
monthly_avg_sales.plot(kind='bar')
plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
plt.tight_layout()
plt.show()

# Average sales by store
store_avg_sales = train_df.groupby('store')['sales'].mean()

# Plot
plt.figure(figsize=(10, 5))
store_avg_sales.plot(kind='bar')
plt.title("Average Sales per Store")
plt.xlabel("Store")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.show()

# Average sales by item
item_avg_sales = train_df.groupby('item')['sales'].mean()

# Plot
plt.figure(figsize=(12, 5))
item_avg_sales.plot(kind='bar')
plt.title("Average Sales per Item")
plt.xlabel("Item")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.show()


# Create time-based features
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek
train_df['is_weekend'] = train_df['dayofweek'].isin([5, 6]).astype(int)
print(train_df[['date', 'year', 'month', 'day', 'dayofweek', 'is_weekend']].head())


# Load test.csv
test_df = pd.read_csv(r"C:\Users\ADITYA\OneDrive\Desktop\New folder\test.csv")
# Convert 'date' to datetime
test_df['date'] = pd.to_datetime(test_df['date'])
# Create same time-based features
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek
test_df['is_weekend'] = test_df['dayofweek'].isin([5, 6]).astype(int)
# Check result
print(test_df[['date', 'year', 'month', 'day', 'dayofweek', 'is_weekend']].head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Features to use
features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend']
target = 'sales'

# Full feature set
X = train_df[features]
y = train_df[target]

# Take 10% sample to reduce training time
X_sample = X.sample(frac=0.1, random_state=42)
y_sample = y.loc[X_sample.index]

# Train-test split on the sample
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Define and train the model
print("Training started...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
print("Prediction complete.")
y_pred = model.predict(X_val)
import numpy as np
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {round(rmse, 2)}")


# Prepare test features
test_features = test_df[features]
# Generate predictions on test set
test_predictions = model.predict(test_features)
# Load the sample submission file
submission_df = pd.read_csv(r"C:\Users\ADITYA\OneDrive\Desktop\New folder\sample_submission.csv")
# Insert predictions
submission_df['sales'] = test_predictions
# Save the submission file
submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully.")


# Select columns for dashboard
dashboard_df = train_df[['date', 'store', 'item', 'sales']]
# Save to CSV
dashboard_df.to_csv("dashboard_data.csv", index=False)
print("Dashboard data exported successfully.")
