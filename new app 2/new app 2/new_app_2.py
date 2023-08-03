import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Task 1: Dataset Selection and Loading
# Load the dataset
dataset_path = "D:/Housing.csv"
df = pd.read_csv(dataset_path)

# Display the first few rows of the DataFrame
print(df.head())

# Generate summary statistics
print(df.describe())

# Identify data types
print(df.dtypes)

# Task 2: Data Exploration with Python
# Visualize the data distribution using a histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], kde=True)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()

# Encode categorical variables
le = LabelEncoder()
df['mainroad'] = le.fit_transform(df['mainroad'])
df['guestroom'] = le.fit_transform(df['guestroom'])
df['basement'] = le.fit_transform(df['basement'])
df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
df['airconditioning'] = le.fit_transform(df['airconditioning'])
df['prefarea'] = le.fit_transform(df['prefarea'])
df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

# Visualize the correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Task 3: Data Preprocessing with Python
# Separate features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 4: Implement Machine Learning Models with Python
# Train and evaluate SVM model
svm_model = SVR()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)
print("SVM Mean Squared Error:", mse_svm)
print("SVM R-squared:", r2_svm)

# Train and evaluate Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest R-squared:", r2_rf)


# Task 5: Visualization with Python

# Add a box plot to visualize the distribution of prices
plt.figure(figsize=(8, 6))
sns.boxplot(x='price', data=df)
plt.xlabel('Price')
plt.title('Box Plot of House Prices')
plt.show()


# Task 6: Documentation for Python Code
# Provided detailed explanations and comments for each step of the analysis


