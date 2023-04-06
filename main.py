import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset into a pandas DataFrame
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Print the first 5 rows of the DataFrame
print(df.head().to_string())

# Print the shape of the DataFrame (number of rows and columns)
print(df.shape)

# Check for missing values in the DataFrame
print(df.isna().sum())
#Found 201 NULL values in bmi column

# Get summary statistics for the numerical columns in the DataFrame
print(df.describe())

# Get information about the DataFrame, including column data types and number of non-null values
print(df.info())

# Print the data types of each column in the DataFrame
print(df.dtypes)

# We must check that there are no unexpected unique values in each column
for col in df.columns:
  if df[col].dtype != 'float64':
    print(f"{col} has unique values:{df[col].unique()}")
#Filling Null Data
#We will fill the null values of bmi column with the mean of this column
df['bmi'].fillna(value=df['bmi'].mean(),inplace=True)
print("after filling null values:",df.isna().sum())


# Convert categorical variables into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

# Normalize the numerical variables
df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['avg_glucose_level'] = (df['avg_glucose_level'] - df['avg_glucose_level'].min()) / (df['avg_glucose_level'].max() - df['avg_glucose_level'].min())
df['bmi'] = (df['bmi'] - df['bmi'].min()) / (df['bmi'].max() - df['bmi'].min())

# Drop the ID column as it is not relevant for modeling
df.drop('id', axis=1, inplace=True)

# Convert the stroke column to integers (0 for no stroke, 1 for stroke)
df['stroke'] = df['stroke'].astype(int)

#PRE PROCESSING + EDA

# Plot a histogram of the age variable
plt.hist(df['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
#Residence Type, Average Glucose Level, BMI, and Smoking Status

# Plot a bar chart of the stroke variable
sns.countplot(x='stroke', data=df)
plt.title('Stroke Counts')
plt.xlabel('Stroke (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# Plot a scatterplot of the age and bmi variables by stroke
sns.scatterplot(x='age', y='bmi', hue='stroke', data=df)
plt.title('Age and BMI by Stroke')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

