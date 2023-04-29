import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import os
from imblearn.over_sampling import SMOTE
import imblearn
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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

# # Checking Outliers Using Box -Plot - for BMI, AVG Glucose Level
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
sns.boxplot(x='bmi', data=df, ax=axs[0])
axs[0].set_title('BMI Distribution')
sns.boxplot(x='avg_glucose_level', data=df, ax=axs[1])
axs[1].set_title('Average Glucose Level Distribution')
plt.tight_layout()
plt.show()

# Methods for treating outliers in the BMI variable
df['bmi'] = np.log(df['bmi'])

# Re-plot the box plot for BMI to confirm outlier treatment
sns.boxplot(x=df['bmi'])
plt.title('BMI Box Plot (after Log)')
plt.xlabel('BMI')
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


# Label Encoding
categorical_col=['gender','ever_married','work_type','Residence_type','smoking_status']
le = LabelEncoder()
for col in categorical_col:
  df[col] = le.fit_transform(df[col])

for col in df.columns:
  if df[col].dtype != 'float64':
    print(f"{col} has unique values:{df[col].unique()}")

print(df.head().to_string())

## Feature Engineering:
over = SMOTE(sampling_strategy=1)
under = RandomUnderSampler(sampling_strategy=0.1)

features = df.loc[:, :'smoking_status']
target = df['stroke']
steps = [('under', under), ('over', over)]
pipeline = Pipeline(steps=steps)
features, target = pipeline.fit_resample(features, target)

print(Counter(target))



# SPLIT TEST AND TRAIN PART
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target, test_size=0.2, random_state=42)

# MACHINE LEARNING ALGORITHMS


# RANDOM FOREST
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("random-forest confusion matrix \n",confusion_matrix(y_test, y_pred))
print("random-forest Classification report \n",classification_report(y_test, y_pred))
print("Accuracy-Random-forest\n:", accuracy)
y_pred_proba = rfc.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='Random Forest (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()


# Gradient Boosting Classifier

gbc = GradientBoostingClassifier(random_state=42)
# Fit the model to the training data
gbc.fit(X_train, y_train)
# Use the model to make predictions on the testing data
y_pred = gbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy-Gradient Boosting Classifier\n:", accuracy)
print("Gradient Boosting Classifier confusion matrix- \n",confusion_matrix(y_test, y_pred))
print("Gradient Boosting Classifier Classification report \n",classification_report(y_test, y_pred))
y_pred_proba = gbc.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='Gradient Boosting Classifier (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()

# XGBOOST
xgb = XGBClassifier()
# Train the classifier on the training data
xgb.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = xgb.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("XGBOOST confusion matrix- \n",confusion_matrix(y_test, y_pred))
print("XGBOOST Classification report  \n",classification_report(y_test, y_pred))
print("Accuracy-XGBOOST\n:", accuracy)
y_pred_proba = xgb.predict_proba(X_test)[:, 1]
# Calculate the AUC score for the XGBoost classifier
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)
# Calculate the false positive rate and true positive rate for various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# Plot the ROC curve
plt.plot(fpr, tpr, label='XGBoost (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()

