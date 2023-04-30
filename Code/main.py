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
# over = SMOTE(sampling_strategy=1)
# under = RandomUnderSampler(sampling_strategy=0.1)

# features = df.loc[:, :'smoking_status']
# target = df['stroke']
# steps = [('under', under), ('over', over)]
# pipeline = Pipeline(steps=steps)
# features, target = pipeline.fit_resample(features, target)

# print(Counter(target))



# # SPLIT TEST AND TRAIN PART
# X_train, X_test, y_train, y_test = train_test_split(features,
#                                                     target, test_size=0.2, random_state=42)
# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
## Feature Engineering:
X = df.drop(['stroke'], axis=1)
y = df['stroke']
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
# # SPLIT TEST AND TRAIN PART
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

# Perform feature scaling using StandardScaler for MLPClassifer and Keras
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Neural Network
from sklearn.neural_network import MLPClassifier
# Neural network model
mlp = MLPClassifier(max_iter=1000, random_state=1)
parameter_space = {
    'hidden_layer_sizes': [(30,5)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.9],
    'learning_rate': ['constant','adaptive', 'learning'],
}
# experimented with these parameters below
# parameter_space = {
#     'hidden_layer_sizes': [(20,5),(5,),(10,),(50,5),(30,17),(30,15),(30,10), (35,10),(20,20)],
#     'activation': ['logistic', 'tanh', 'relu'],
#     'solver': ['sgd', 'adam', 'lbfgs'],
#     'alpha': [0.9, 0.99, 0.1, 0.0001, 0.05, 0.5],
#     'learning_rate': ['constant','adaptive', 'learning'],
# }

# GridSearch to find best parameters
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train_scaled, y_train)
train_acc = clf.score(X_train_scaled, y_train)
test_acc = clf.score(X_test_scaled, y_test)

print(f'Train accuracy: {train_acc:.3f}')
print(f'Test accuracy: {test_acc:.3f}')
print('Best parameters found:\n', clf.best_params_)

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


# Keras Model
from keras.models import Sequential
from keras.layers import Dense

# model architecture - simple model
model = Sequential()
model.add(Dense(30, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=300)
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

#Models
# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

X = df.drop(['stroke'], axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('K-Nearest Neighbors Accuracy:', accuracy_score(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print('Support Vector Machine Accuracy:', accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Create a list of tuples containing model names and instances
models = [('Logistic Regression', lr), ('KNN', knn), ('SVM', svm)]

# Iterate over the models
for name, model in models:
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # Compute the confusion matrix
    conf_mat = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    # Set the title of the plot
    plt.title(f"{name} Confusion Matrix")
    # Add a color bar to the plot
    plt.colorbar(conf_mat.im_, ax=plt.gca())
    # Show the plot
    plt.show()
from sklearn.metrics import plot_roc_curve

# Create a logistic regression model
logreg = LogisticRegression()
# Train the model on the training data
logreg.fit(X_train, y_train)
# Plot the ROC curve
plot_roc_curve(logreg, X_test, y_test)
# Set the title of the plot
plt.title("Logistic Regression ROC Curve")
# Show the plot
plt.show()


from sklearn.metrics import plot_precision_recall_curve

# Create a logistic regression model
logreg = LogisticRegression()
# Train the model on the training data
logreg.fit(X_train, y_train)
# Plot the precision-recall curve
plot_precision_recall_curve(logreg, X_test, y_test)
# Set the title of the plot
plt.title("Logistic Regression Precision-Recall Curve")
# Show the plot
plt.show()

# Create a logistic regression model
logreg = LogisticRegression()
# Train the model on the training data
logreg.fit(X_train, y_train)
# Get the coefficients of the model
coefs = logreg.coef_[0]
# Create a list of feature names
features = X.columns
# Create a bar chart of the feature importances
plt.bar(features, coefs)
# Set the title of the plot
plt.title("Logistic Regression Feature Importances")
# Show the plot
plt.show()

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

# Scale the test data using the same scaler used for training data
X_test_scaled = scaler.transform(X_test)

# Confusion Matrix for KNN
plot_confusion_matrix(knn, X_test_scaled, y_test, cmap='Blues')
plt.title('KNN Confusion Matrix')

# ROC Curve for KNN
plot_roc_curve(knn, X_test_scaled, y_test)
plt.title('KNN ROC Curve')

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

# Scale the test data using the same scaler used for training data
X_test_scaled = scaler.transform(X_test)

# Confusion Matrix for SVM
plot_confusion_matrix(svm, X_test_scaled, y_test, cmap='Blues')
plt.title('SVM Confusion Matrix')

# ROC Curve for SVM
plot_roc_curve(svm, X_test_scaled, y_test)
plt.title('SVM ROC Curve')


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_score = dt.score(X_test_scaled, y_test)
print('Decision Tree Accuracy:', dt_score)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)
print('Random Forest Accuracy:', rf_score)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)
gb_score = gb.score(X_test_scaled, y_test)
print('Gradient Boosting Accuracy:', gb_score)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_score = nb.score(X_test_scaled, y_test)
print('Naive Bayes Accuracy:', nb_score)

# Multi-Layer Perceptron
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_score = mlp.score(X_test_scaled, y_test)
print('MLP Accuracy:', mlp_score)

