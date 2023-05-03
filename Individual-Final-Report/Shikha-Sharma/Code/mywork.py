import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

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
#
# Smoking status Analysis
print("Value of count of smoking status-\n",df['smoking_status'].value_counts())
# Plot a bar chart of stroke cases by smoking status -pie
df['smoking_status'].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.title('Distribution of Smoking Status')
plt.axis('equal')
plt.tight_layout()
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
nums = list(df.select_dtypes(include=['int64','float64']))
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for i in range(0, len(nums)):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df[nums[i]],color='green',orient='v')
    plt.tight_layout()
# Methods for treating outliers in the BMI variable
outlier = ['avg_glucose_level', 'bmi']
Q1 = df[outlier].quantile(0.25)
Q3 = df[outlier].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[outlier]<(Q1-1.5*IQR))|(df[outlier]>(Q3+1.5*IQR))).any(axis=1)]
df.reset_index(drop=True)
plt.figure(figsize=(20, 10))
for i in range(0, len(nums)):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df[nums[i]],color='green',orient='v')
    plt.tight_layout()
plt.show()

# Label Encoding
categorical_col=['gender','ever_married','work_type','Residence_type','smoking_status']
le = LabelEncoder()
for col in categorical_col:
  df[col] = le.fit_transform(df[col])

# Feature Engineering:
X = df.drop(['stroke'], axis=1)
y = df['stroke']

# # SPLIT TEST AND TRAIN PART
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plt.figure(figsize=(10, 5))
plt.title("Class Distribution before SMOTE")
plt.hist(y, bins=2, rwidth=0.8)
plt.xticks([0, 1])
plt.xlabel("Stroke(0=No,1=Yes)")
plt.ylabel("Count")
plt.show()
unique, counts = np.unique(y_train, return_counts=True)

# Print the count of instances in each class before oversampling
print("Class counts before SMOTE oversampling:")
for i in range(len(unique)):
    print("Class", unique[i], ":", counts[i])

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Count the number of instances in each class after oversampling
unique, counts = np.unique(y_train, return_counts=True)

# Print the count of instances in each class after oversampling
print("Class counts after SMOTE oversampling:")
for i in range(len(unique)):
    print("Class", unique[i], ":", counts[i])
plt.figure(figsize=(10, 5))
plt.title("Class Distribution after SMOTE")
plt.hist(y_train, bins=2, rwidth=0.8)
plt.xticks([0, 1])
plt.xlabel("Stroke(0=No,1=Yes)")
plt.ylabel("Count")
plt.legend()
plt.show()
# Perform feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# MACHINE LEARNING ALGORITHMS
# RANDOM FOREST
rfc = RandomForestClassifier(random_state=42)
scores = cross_val_score(rfc, X_train_scaled, y_train, cv=10)
rfc.fit(X_train_scaled, y_train)
y_pred = rfc.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("random-forest confusion matrix \n",confusion_matrix(y_test, y_pred))
print("random-forest Classification report \n",classification_report(y_test, y_pred))
print("Accuracy-Random-forest\n:", accuracy)
y_pred_proba = rfc.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='Random Forest (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()
# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
scores = cross_val_score(gbc, X_train_scaled, y_train, cv=10)
print(scores)
# Fit the model to the training data
gbc.fit(X_train_scaled, y_train)
# Use the model to make predictions on the testing data
y_pred = gbc.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy-Gradient Boosting Classifier\n:", accuracy)
print("Gradient Boosting Classifier confusion matrix- \n",confusion_matrix(y_test, y_pred))
print("Gradient Boosting Classifier Classification report \n",classification_report(y_test, y_pred))
y_pred_proba = gbc.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='Gradient Boosting Classifier (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()
# XGBOOST
xgb = XGBClassifier()
scores = cross_val_score(xgb, X_train_scaled, y_train, cv=10)
# Train the classifier on the training data
xgb.fit(X_train_scaled, y_train)
# Make predictions on the testing data
y_pred = xgb.predict(X_test_scaled)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("XGBOOST confusion matrix- \n",confusion_matrix(y_test, y_pred))
print("XGBOOST Classification report  \n",classification_report(y_test, y_pred))
print("Accuracy-XGBOOST\n:", accuracy)
y_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]
# Calculate the AUC score for the XGBoost classifier
auc = roc_auc_score(y_test, y_pred_proba)
# Calculate the false positive rate and true positive rate for various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# Plot the ROC curve
plt.plot(fpr, tpr, label='XGBoost (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()
#Cross Validation
#Random Forest
rfc = RandomForestClassifier(random_state=42)
rfc_scores = cross_val_score(rfc, X_train_scaled, y_train, cv=10)
print('Random Forest Accuracy:', rfc_scores.mean())
#Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc_scores = cross_val_score(gbc, X_train_scaled, y_train, cv=10)
print('Gradient Boosting Classifier Accuracy:', gbc_scores.mean())
# XGB
xgb = XGBClassifier()
xgb_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=10)
print('XGBoost Accuracy:', xgb_scores.mean())

