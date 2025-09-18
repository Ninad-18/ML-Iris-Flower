# Import necessary libraries
import pandas as pd                             # For loading and handling tabular data
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
import numpy as np                               # For creating and manipulating arrays
from sklearn.model_selection import train_test_split # To split data into training and test sets
from sklearn.preprocessing import StandardScaler     # For feature scaling (standardization)
from sklearn.metrics import accuracy_score           # To measure model accuracy

# ------------------- Load and explore the dataset -------------------

# Load the iris dataset from a CSV file
iris_data = pd.read_csv('iris_dataset.csv')

# Display the first few rows to inspect the data
iris_data.head()

# ------------------- Prepare features and target --------------------

# Separate the features (X) and the target variable (y)
# Drop 'Id' and 'Species' columns from features; 'Species' is the target.
X = iris_data.drop(columns=['target'])
y = iris_data['target']

# Quick look at the feature data
X.head()

# ------------------- Split the dataset -----------------------------

# Split the dataset into training and testing sets (80% train, 20% test)
# random_state ensures the split is reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- Feature scaling -------------------------------

# Standardize features: subtract mean and scale to unit variance
scaler = StandardScaler()

# Fit scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler (do not fit again!)
X_test_scaled = scaler.transform(X_test)

# ------------------- Train the Logistic Regression model -----------

# Create a Logistic Regression model instance
model = LogisticRegression()

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

# ------------------- Make predictions and evaluate ------------------

# Predict the species for the scaled test set
y_pred = model.predict(X_test_scaled)

# Calculate and display the accuracy of the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ------------------- Predict on new unseen data ---------------------

# Create a NumPy array of new flower measurements (each row is one flower)
new_data = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Example 1
    [6.3, 2.9, 5.6, 1.8],  # Example 2
    [4.9, 3.0, 1.4, 0.2]   # Example 3
])

# Scale the new data using the previously fitted scaler
new_data_scaled = scaler.transform(new_data)

# Predict the species for these new samples
predictions = model.predict(new_data_scaled)

# Display the predicted species
print("Predictions:", predictions)
