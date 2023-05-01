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
#%%
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
# Plotting age histogram
plt.hist(df['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()
# Plotting age histogram for strokes
plt.hist(df[df['stroke'] == 1]['age'],bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution for Stroke Cases')
plt.show()
# Plotting heart disease bar chart
heart_disease_counts = df['heart_disease'].value_counts()
plt.bar(heart_disease_counts.index, heart_disease_counts.values)
plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
plt.ylabel('Count')
plt.title('Heart Disease Distribution')
plt.show()
# Plotting heart disease bar chart for strokes
heart_disease_counts = df[df['stroke'] == 1]['heart_disease'].value_counts()
plt.bar(heart_disease_counts.index, heart_disease_counts.values)
plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
plt.ylabel('Count')
plt.title('Heart Disease Distribution for Stroke Cases')
plt.show()
# Plotting age vs heart disease for strokes
plt.scatter(df[df['stroke'] == 1]['age'], df[df['stroke'] == 1]['heart_disease'],
            label='Stroke', color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Heart Disease')
plt.title('Age vs Heart Disease for Stroke Cases')
plt.legend()
plt.show()
# Plotting ever married pie chart
fig = plt.figure(facecolor='white')
ever_married_counts = df['ever_married'].value_counts()
plt.pie(ever_married_counts.values, labels=ever_married_counts.index, autopct='%1.1f%%')
plt.title('Ever Married Distribution')
plt.show()
# Plotting ever married pie chart for strokes
fig = plt.figure(facecolor='white')
ever_married_counts = df[df['stroke'] == 1]['ever_married'].value_counts()
plt.pie(ever_married_counts.values, labels=ever_married_counts.index, autopct='%1.1f%%')
plt.title('Ever Married Distribution for Stroke Cases')
plt.show()
# Plotting age vs ever married for strokes
plt.scatter(df[df['stroke'] == 1]['age'], df[df['stroke'] == 1]['ever_married'],
            label='Stroke', color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Ever Married')
plt.title('Age vs Ever Married for Stroke Cases')
plt.legend()
plt.show()
# Plotting ever married vs heart disease for strokes
ever_married_counts = df[df['stroke'] == 1]['ever_married'].value_counts()
heart_disease_counts = df[df['stroke'] == 1]['heart_disease'].value_counts()
plt.bar(['Not Ever Married, No Heart Disease', 'Not Ever Married, Heart Disease',
         'Ever Married, No Heart Disease', 'Ever Married, Heart Disease'],
        [df[(df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['stroke'] == 1)].shape[0],
         df[(df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['stroke'] == 1)].shape[0],
         df[(df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['stroke'] == 1)].shape[0],
         df[(df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['stroke'] == 1)].shape[0]],
        color=['green', 'red', 'purple', 'blue'], alpha=0.5)
plt.ylabel('Count')
plt.title('Ever Married vs Heart Disease Distribution for Stroke Cases')
plt.xticks(rotation=45)
plt.show()
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
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
plot_confusion_matrix(rfc, X_test_scaled, y_test, cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plot_roc_curve(rfc, X_test_scaled, y_test)
plt.title('Random Forest ROC Curve')
plot_confusion_matrix(gbc, X_test_scaled, y_test, cmap='Blues')
plt.title('Gradient Boosting Classifier Confusion Matrix')
plot_roc_curve(gbc, X_test_scaled, y_test)
plt.title('Gradient Boosting Classifier ROC Curve')
plot_confusion_matrix(xgb, X_test_scaled, y_test, cmap='Blues')
plt.title('XGBOOST Confusion Matrix')
plot_roc_curve(xgb, X_test_scaled, y_test)
plt.title('XGBOOST ROC Curve')
#%%
#Cross Validation
#Random Forest
rfc = RandomForestClassifier(random_state=42)
rfc_scores = cross_val_score(rfc, X_train_scaled, y_train, cv=10)
print('Random Forest Accuracy:', rfc_scores.mean())
#Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc_scores = cross_val_score(gbc, X_train_scaled, y_train, cv=10)
print('Gradient Boosting Classifier Accuracy:', gbc_scores.mean())
# Support Vector Machine
xgb = XGBClassifier()
xgb_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=10)
print('XGBoost Accuracy:', xgb_scores.mean())


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
np.random.seed(123)
# Initialize a Random Forest Classifier and a Recursive Feature Elimination with Cross Validation selector
estimator = RandomForestClassifier()
selector = RFECV(estimator=estimator, step=1, cv=5, scoring='recall')
# Fit the selector to the scaled training data
selector.fit(X_train_scaled, y_train)
# Select the important features from the training and test data using the selector
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Train a new model using the selected features with a Multi-layer Perceptron Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2,3,2), activation='relu', solver='adam', alpha=0.00001, learning_rate='adaptive')
mlp.fit(X_train_selected, y_train)
# Make predictions on the test data using the trained model
y_pred = mlp.predict(X_test_selected)
# Evaluate Model with Metrics Below
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
print(f'F1-score: {f1_score(y_test, y_pred):.3f}')

#  ROC AUC Metric
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# Calculate the AUC score
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#%%
# Neural network model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)
np.random.seed(2)
# Create parameter space for Grid Serch to run and find Best Params
parameter_space = {
    'hidden_layer_sizes': [(80,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [.5],
    'learning_rate': ['adaptive','constant', 'learning'],
}
# experimented with these parameters below
# parameter_space = {
#     'hidden_layer_sizes': [(20,5),(5,),(10,),(50,5)],
#     'activation': ['logistic', 'tanh', 'relu'],
#     'solver': ['sgd', 'adam', 'lbfgs'],
#     'alpha': [0.9, 0.99, 0.1, 0.0001, 0.05, 0.5],
#     'learning_rate': ['constant','adaptive', 'learning'],
# }
# GridSearch to find best parameters
from sklearn.model_selection import GridSearchCV
# Tune Params and fit
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, scoring='recall')
clf.fit(X_train_scaled, y_train)

# Score Train and Test Data
train_acc = clf.score(X_train_scaled, y_train)
test_acc = clf.score(X_test_scaled, y_test)

print(f'Train accuracy: {train_acc:.3f}')
print(f'Test accuracy: {test_acc:.3f}')
print('Best parameters found:\n', clf.best_params_)
from sklearn.model_selection import cross_val_score

y_pred = clf.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
print(f'F1-score: {f1_score(y_test, y_pred):.3f}')
#  ROC AUC Metric
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# Calculate the AUC score
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
#%%
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Define the pipeline
pipeline = Pipeline(steps=[
    ('pca', PCA()),
    ('mlp', MLPClassifier(max_iter=400))
])

# Define the parameter space
parameter_space = {
    'pca__n_components': [1],
    # 'mlp__hidden_layer_sizes': [(2,2,2,2), (3,3,3), (2,3,2)],
    'mlp__hidden_layer_sizes': [(2,3,2)],
    'mlp__activation': ['relu', 'logistic', 'tanh'],
    'mlp__solver': ['adam', 'sgd'],
    # 'mlp__alpha': [.01, 0.009, 0.05, 0.001],
    'mlp__alpha': [0.01],
    'mlp__learning_rate': ['adaptive', 'learning'],
}

# Create the grid search object
clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=5, scoring='recall')

# Fit the grid search object to the data
clf.fit(X_train_scaled, y_train)
best_model = clf.best_estimator_
best_param = clf.best_params_
print(best_model)
print(best_param)

# Make predictions with the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model performance
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
print(f'F1-score: {f1_score(y_test, y_pred):.3f}')


#%%
#  ROC AUC Metric
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# Calculate the AUC score
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#%%

# Keras Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
# model architecture - simple model
model = Sequential()
model.add(Dense(30, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50,verbose=0)
# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')
# Plot the accuracy and loss curves for each epoch
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Training and validation accuracy and loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()
# Predict probabilities for test set
from sklearn.metrics import auc
y_pred_keras = model.predict(X_test_scaled)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
# Model evaluation
y_pred_keras_binary = (y_pred_keras > 0.5).astype(int) # threshold at 0.5
print(f'Accuracy: {accuracy_score(y_test, y_pred_keras_binary):.3f}')
print(f'Precision: {precision_score(y_test, y_pred_keras_binary):.3f}')
print(f'Recall: {recall_score(y_test, y_pred_keras_binary):.3f}')
print(f'F1-score: {f1_score(y_test, y_pred_keras_binary):.3f}')


#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

model = Sequential()

# Input layer
model.add(Dense(units=256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

y_pred_k = model.predict(X_test_scaled)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
# Model evaluation
y_pred_k = (y_pred_k > 0.5).astype(int) # threshold at 0.5
print(f'Accuracy: {accuracy_score(y_test, y_pred_k):.3f}')
print(f'Precision: {precision_score(y_test, y_pred_k):.3f}')
print(f'Recall: {recall_score(y_test, y_pred_k):.3f}')
print(f'F1-score: {f1_score(y_test, y_pred_k):.3f}')



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
