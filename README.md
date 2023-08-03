# Lab10
The objective of this assignment is to explore advanced Python tools for machine learning. Students will utilize publicly available datasets from the provided GitHub repositories to perform data exploration, preprocessing, implement machine learning models, and visualize the results using Python programming only


# Code Descriptions for 
# DCEU Box Office and Rating Dataset 1

## Task 2: Data Exploration with Python

This section explores the DCEU (DC Extended Universe) Box Office and Rating dataset and visualizes the distribution of the 'audience_score' feature.

1. Load the dataset: The dataset is loaded from the CSV file "dceu_box_office_and_rating.csv" using pandas.

2. Display the first few rows: The first few rows of the DataFrame are displayed using the `head()` function to get a quick overview of the data.

3. Generate summary statistics: The summary statistics of the dataset are printed using the `describe()` function to get insights into the data.

4. Identify data types: The data types of each column in the DataFrame are printed using the `dtypes` attribute.

5. Visualize the data distribution: A histogram is plotted for the 'audience_score' feature using seaborn's `histplot()` function.

## Task 3: Data Preprocessing with Python

This section prepares the data for training the machine learning models.

1. Define X and y: The features (X) and the target variable (y) are defined. We drop the columns 'movie_title', 'release_date', and 'worldwide_gross ($)' from the features as they are not required for prediction.

2. Split the data: The data is split into training and testing sets using the `train_test_split()` function from scikit-learn.

## Task 4: Implement Machine Learning Models with Python

This section implements two machine learning models, SVM and Random Forest, and evaluates their performance.

1. Train and evaluate SVM model: The Support Vector Machine (SVM) model is trained using scikit-learn's `SVC()` class. The accuracy of the SVM model is calculated using the `accuracy_score()` function, and the classification report is printed.

2. Train and evaluate Random Forest model: The Random Forest model is trained using scikit-learn's `RandomForestClassifier()` class. The accuracy of the Random Forest model is calculated using the `accuracy_score()` function, and the classification report is printed.

## Task 5: Visualization with Python

This section includes visualizations to gain insights from the data.

1. Box plot of audience_score based on US Gross: A box plot is plotted to visualize the distribution of 'audience_score' based on the 'US_gross ($)' feature.

2. Line chart of audience_score over IMDB Ratings: A line chart is plotted to visualize the distribution of 'audience_score' over the 'imdb' feature.

## Task 6: Documentation for Python Code

This section provides detailed explanations and comments for each step of the analysis.
 

## Code Descriptions for 
# Housing Price Prediction Dataset 2 with Machine Learning

This repository contains Python code that explores a housing dataset, preprocesses the data, and implements two machine learning models (Support Vector Machine and Random Forest) to predict housing prices. The dataset used in this analysis is stored in the "Housing.csv" file.

## Task 1: Dataset Selection and Loading

The housing dataset is loaded from the "Housing.csv" file using the pandas library. The first few rows and summary statistics of the dataset are displayed to understand the data better.

## Task 2: Data Exploration with Python

The distribution of house prices is visualized using a histogram. Categorical variables are encoded using LabelEncoder for further analysis.

## Task 3: Data Preprocessing with Python

The features and the target variable are separated, and the data is split into training and testing sets using the train_test_split function from scikit-learn.

## Task 4: Implement Machine Learning Models with Python

Two machine learning models, Support Vector Machine (SVM) and Random Forest, are implemented and trained using scikit-learn. The accuracy of the models is evaluated using the Mean Squared Error (MSE) and R-squared metrics.

## Task 5: Visualization with Python

Box plots are used to visualize the distribution of house prices, and a heatmap is plotted to show the correlation between features.

## Task 6: Documentation for Python Code

The Python code includes detailed explanations and comments for each step of the analysis.

### Requirements

The code in this repository requires the following Python libraries:

- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

