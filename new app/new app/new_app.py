import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Task 2: Data Exploration with Python
# Load the dataset
dataset_path = "D:\dceu_box_office_and_rating.csv"
df = pd.read_csv(dataset_path)

# Display the first few rows of the DataFrame
print(df.head())

# Generate summary statistics
print(df.describe())

# Identify data types
print(df.dtypes)

# Visualize the data distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['audience_score'], kde=True)
plt.xlabel('Audience Score')
plt.ylabel('Frequency')
plt.title('Distribution of Audience Scores')
plt.show()

# Task 3: Data Preprocessing with Python
# Define X and y
X = df.drop(['movie_title', 'release_date', 'worldwide_gross ($)'], axis=1)  # Features
y = df['worldwide_gross ($)']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 4: Implement Machine Learning Models with Python
# Train and evaluate SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print(classification_report(y_test, y_pred_svm, zero_division=1.0))  # Set zero_division to 1.0

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print(classification_report(y_test, y_pred_rf, zero_division=1.0))  # Set zero_division to 1.0

# Task 5: Visualization with Python
# Visualizing the distribution of audience_score based on the 'US_gross ($)' feature using a box plot
plt.figure(figsize=(10, 8))
sns.boxplot(x='US_gross ($)', y='audience_score', data=df)
plt.xlabel('US Gross ($)')
plt.ylabel('Audience Score')
plt.title('Audience Score Distribution based on US Gross')
plt.show()

# Visualize the distribution of 'audience_score' over the 'imdb' feature using a line chart
plt.figure(figsize=(10, 6))
plt.plot(df['imdb'], df['audience_score'], marker='o', linestyle='-', color='blue')
plt.xlabel('IMDB Rating')
plt.ylabel('Audience Score')
plt.title('Distribution of Audience Scores over IMDB Ratings')
plt.show()

# Task 6: Documentation for Python Code
# Provided detailed explanations and comments for each step of the analysis

