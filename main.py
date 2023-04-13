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
# Found 201 NULL values in bmi column

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


# Filling Null Data
# We will fill the null values of bmi column with the mean of this column
df['bmi'].fillna(value=df['bmi'].mean(),inplace=True)
print("after filling null values:",df.isna().sum())

# Drop the ID column as it is not relevant for modeling
df.drop('id', axis=1, inplace=True)

# Convert the stroke column to integers (0 for no stroke, 1 for stroke)
df['stroke'] = df['stroke'].astype(int)

# PRE PROCESSING + EDA

# Target feature - Stroke
print("Value count in the stroke : \n",df['stroke'].value_counts())

# Plot a bar chart of the stroke variable
sns.countplot(x='stroke', data=df)
plt.title('Stroke Counts')
plt.xlabel('Stroke (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# Smoking status Analysis
print("Value of count of smoking status-\n",df['smoking_status'].value_counts())

# Plot a bar chart of stroke cases by smoking status -pie
df['smoking_status'].value_counts().plot(kind="pie")
plt.title('Smoking Status')
plt.ylabel('')
plt.show()

# Bar chart
sns.countplot(x='smoking_status', hue='stroke', data=df)
plt.title('Stroke Cases by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.show()

# #Residence Type:
print("Value of count of residence-\n",df['Residence_type'].value_counts())
# Plot a bar chart of stroke cases by residence type
sns.countplot(x='Residence_type', hue='stroke', data=df)
plt.title('Stroke Cases by Residence Type')
plt.xlabel('Residence Type')
plt.ylabel('Count')
plt.show()

# Average Glucose Level:
sns.regplot(x='avg_glucose_level', y='stroke', data=df, logistic=True)
plt.title('Relationship between Average Glucose Level and Stroke')
plt.xlabel('Average Glucose Level')
plt.ylabel('Stroke (0=No, 1=Yes)')
plt.show()

# # Plot histograms of glucose levels for those with and without stroke
plt.hist(df[df['stroke'] == 0]['avg_glucose_level'], alpha=0.5, label='No Stroke', bins=20)
plt.hist(df[df['stroke'] == 1]['avg_glucose_level'], alpha=0.5, label='Stroke', bins=20)
plt.title('Glucose Levels by Stroke Status')
plt.xlabel('Average Glucose Level')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot histogram of glucose levels for entire dataset
sns.histplot(data=df['avg_glucose_level'])
plt.title('Glucose Level Distribution')
plt.xlabel('Average Glucose Level')
plt.ylabel('Frequency')
plt.show()

# BMI
sns.regplot(x='bmi', y='stroke', data=df, logistic=True)
plt.title('Relationship between BMI and Stroke')
plt.xlabel('BMI')
plt.ylabel('Stroke (0=No, 1=Yes)')
plt.show()

# Plot histograms of BMI for those with and without stroke
plt.hist(df[df['stroke'] == 0]['bmi'], alpha=0.5, label='No Stroke', bins=20)
plt.hist(df[df['stroke'] == 1]['bmi'], alpha=0.5, label='Stroke', bins=20)
plt.title('BMI by Stroke Status')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot  BMI for entire dataset
sns.histplot(data=df['bmi'])
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

#EDA by Sanjana
#
#Heat map
sns.heatmap(df.corr())

#Correlation plot
df.corr()

#Stacked Histogram of Gender and Stroke
sns.histplot(data=df, x='gender',hue='stroke',
    multiple="stack",
    palette="rocket",
    edgecolor=".3",
    linewidth=.5)

#Stacked Histogram of hypertension and Stroke
sns.histplot(data=df, x='hypertension',hue='stroke',
    bins=[-0.5, 0.5, 1.5], 
    discrete=True,
    multiple="stack",
    palette="rocket",
    edgecolor=".3",
    linewidth=.5)
plt.xticks([0, 1])
plt.show()

#Stacked Histogram of Work type and Stroke
sns.histplot(data=df, x='work_type',hue='stroke',
    multiple="stack",
    palette="rocket",
    edgecolor=".3",
    linewidth=.5)

#Box Plot of hypertension vs age
sns.boxplot(x='hypertension', y='age', data=df)
plt.show()

#Scatter plot of Hypertention vs age
sns.scatterplot(x='age', y='hypertension', data=df)
plt.show()

#Pair plot for age, avg glucose level, bmi and stroke
sns.pairplot(df[['age', 'avg_glucose_level', 'bmi', 'stroke']])
plt.show()
