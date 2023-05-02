#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import os
from imblearn.over_sampling import SMOTE
import imblearn
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
#%%
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
#%%
#Stacked Histogram of Gender and Stroke
sns.histplot(data=df, x='gender',hue='stroke',
    multiple="stack",
    palette="rocket",
    edgecolor=".3",
    linewidth=.5)
plt.show()
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
plt.show()
#Box Plot of hypertension vs age
sns.boxplot(x='hypertension', y='age', data=df)
plt.show()
#Scatter plot of Hypertention vs age
sns.scatterplot(x='age', y='hypertension', data=df)
plt.show()
#Pair plot for age, avg glucose level, bmi and stroke
sns.pairplot(df[['age', 'avg_glucose_level', 'bmi', 'stroke']])
plt.show()
#%%
# Label Encoding
categorical_col=['gender','ever_married','work_type','Residence_type','smoking_status']
le = LabelEncoder()
for col in categorical_col:
  df[col] = le.fit_transform(df[col])
#%%
# HeatMap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation map for variables')
plt.tight_layout()
plt.show()
correlation = df.corr()
print(correlation)
#%%
# Encode categorical variables
df = pd.get_dummies(df, columns=['gender','hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type','smoking_status'])

#%%
# Feature Engineering:
df = df[['age', 'hypertension_0', 'hypertension_1', 'heart_disease_0', 'heart_disease_1', 'stroke']]
# df = df[['age', 'hypertension', 'heart_disease', 'stroke']]
X = df.drop(['stroke'], axis=1)

y = df['stroke']
# HeatMap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation map for variables')
plt.tight_layout()
plt.show()
correlation = df.corr()
print(correlation)
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
#%%
# Print the count of instances in each class before oversampling
print("Class counts before SMOTE oversampling:")
for i in range(len(unique)):
    print("Class", unique[i], ":", counts[i])
#%%
# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Count the number of instances in each class after oversampling
unique, counts = np.unique(y_train, return_counts=True)
#%%
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


#%%
# Perform feature scaling 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
# Models
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print('Logistic Regression Accuracy:', accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
print('K-Nearest Neighbors Accuracy:', accuracy_score(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
# Support Vector Machine
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
print('Support Vector Machine Accuracy:', accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
# Create a list of tuples containing model names and instances
models = [('Logistic Regression', lr), ('KNN', knn), ('SVM', svm)]
# Confusion Matrix for Logistic
plot_confusion_matrix(lr, X_test_scaled, y_test, cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
# ROC Curve for Logistic
plot_roc_curve(lr, X_test_scaled, y_test)
plt.title('Logistic Regression ROC Curve')
# Confusion Matrix for KNN
plot_confusion_matrix(knn, X_test_scaled, y_test, cmap='Blues')
plt.title('KNN Confusion Matrix')
# ROC Curve for KNN
plot_roc_curve(knn, X_test_scaled, y_test)
plt.title('KNN ROC Curve')
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
# Confusion Matrix for SVM
plot_confusion_matrix(svm, X_test_scaled, y_test, cmap='Blues')
plt.title('SVM Confusion Matrix')
# ROC Curve for SVM
plot_roc_curve(svm, X_test_scaled, y_test)
plt.title('SVM ROC Curve')
# %%
#Cross Validation
# Logistic Regression
lr = LogisticRegression()
lr_scores = cross_val_score(lr, X_train_scaled, y_train, cv=10)
print('Logistic Regression Accuracy:', lr_scores.mean())
# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10)
print('K-Nearest Neighbors Accuracy:', knn_scores.mean())
# Support Vector Machine
svm = SVC()
svm_scores = cross_val_score(svm, X_train_scaled, y_train, cv=10)
print('Support Vector Machine Accuracy:', svm_scores.mean())