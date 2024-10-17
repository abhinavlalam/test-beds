##################
## 1. Installing the Required Libraries

# pip install pandas scikit-learn matplotlib # Already installed

##################
## 2. Loading the Dataset

import pandas as pd

# Load the dataset
data = pd.read_csv('BostonHousing.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display information about the dataset
print("\nDataset Info:")
print(data.info())

# Describe the dataset (to see statistical information)
print("\nStatistical Summary of the dataset:")
print(data.describe())

##################
## 3. Checking for Missing Values

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

##################
## 4. handle these missing values

"""
# Drop any row that contains missing values using dropna()
# Remove rows with missing values
data_cleaned = data.dropna()

# Check if there are any remaining missing values
print("\nAfter dropping missing values:")
print(data_cleaned.isnull().sum())
"""

# If you decide to impute the missing values using the mean or median, the model's performance will generally not be affected much as long as the percentage of missing data is small

# Fill missing values in the 'rm' column with the mean of the column
data['rm'].fillna(data['rm'].mean(), inplace=True)

# Check if there are any remaining missing values
print("\nAfter filling missing values with the mean:")
print(data.isnull().sum())

##################
## 5. Splitting the Data into Features (X) and Target (y)

# Split the data into features (X) and target (y)
X = data.drop('medv', axis=1)  # Features
y = data['medv']  # Target variable

# Display the features
print("\nFeatures (X):")
print(X.head())

# Display target variable
print("\nTarget (y):")
print(y.head())

##################
## 6. Splitting the Data into Training and Testing Sets:

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"\nTraining set shape (X_train): {X_train.shape}, (y_train): {y_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}, (y_test): {y_test.shape}")

##################
## 7. Training a Decision Tree Model¶

from sklearn.tree import DecisionTreeRegressor
# Create the Decision Tree Model
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Display the first 5 actual vs predicted values
print("\nFirst 5 Actual vs Predicted values:")
for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"Actual: {actual}, Predicted: {predicted}")

##################
## 9. Evaluating the Model

from sklearn.metrics import mean_absolute_error, r2_score
# Calculate MAE and R²
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"\nMean Absolute Error: {mae}")
print(f"R² Score: {r2}")

##################
## 10. Visualizing the Results

import matplotlib.pyplot as plt
# Plot Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()