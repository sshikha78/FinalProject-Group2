#%%
# package install
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# %%
# Load Stroke Preciction Dataset
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke_full_variables = stroke.drop('id', axis=1)
stroke.head()
# %%
# I am focusing on the variables age, heart disease, and married
variables = ['age', 'heart_disease', 'ever_married', 'stroke']

for variable in variables:
    print(variable, stroke[variable].isnull().values.any())
    print(variable, stroke[variable].dtype)
    print()
# For my variables there are no null values

# %%
# Plotting age histogram
plt.hist(stroke['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Plotting age histogram for strokes
plt.hist(stroke[stroke['stroke'] == 1]['age'],bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution for Stroke Cases')
plt.show()

#%%
sns.kdeplot(data=stroke, x='age', hue='stroke', common_norm=False, fill=True, alpha=0.5)
plt.title('Density Chart of Age with Stroke as Hue')
plt.show()


#%%
# Plotting heart disease bar chart
heart_disease_counts = stroke['heart_disease'].value_counts()
plt.bar(heart_disease_counts.index, heart_disease_counts.values)
plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
plt.ylabel('Count')
plt.title('Heart Disease Distribution')
plt.show()

# Plotting heart disease bar chart for strokes
heart_disease_counts = stroke[stroke['stroke'] == 1]['heart_disease'].value_counts()
plt.bar(heart_disease_counts.index, heart_disease_counts.values)
plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
plt.ylabel('Count')
plt.title('Heart Disease Distribution for Stroke Cases')
plt.show()

# Plotting age vs heart disease for strokes
plt.scatter(stroke[stroke['stroke'] == 1]['age'], stroke[stroke['stroke'] == 1]['heart_disease'],
            label='Stroke', color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Heart Disease')
plt.title('Age vs Heart Disease for Stroke Cases')
plt.legend()
plt.show()

#%%
# Plotting ever married pie chart
fig = plt.figure(facecolor='white')
ever_married_counts = stroke['ever_married'].value_counts()
plt.pie(ever_married_counts.values, labels=ever_married_counts.index, autopct='%1.1f%%')
plt.title('Ever Married Distribution')
plt.show()

# Plotting ever married pie chart for strokes
fig = plt.figure(facecolor='white')
ever_married_counts = stroke[stroke['stroke'] == 1]['ever_married'].value_counts()
plt.pie(ever_married_counts.values, labels=ever_married_counts.index, autopct='%1.1f%%')
plt.title('Ever Married Distribution for Stroke Cases')
plt.show()

# Plotting age vs ever married for strokes
plt.scatter(stroke[stroke['stroke'] == 1]['age'], stroke[stroke['stroke'] == 1]['ever_married'],
            label='Stroke', color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Ever Married')
plt.title('Age vs Ever Married for Stroke Cases')
plt.legend()
plt.show()

# Plotting ever married vs heart disease for strokes
ever_married_counts = stroke[stroke['stroke'] == 1]['ever_married'].value_counts()
heart_disease_counts = stroke[stroke['stroke'] == 1]['heart_disease'].value_counts()
plt.bar(['Not Ever Married, No Heart Disease', 'Not Ever Married, Heart Disease',
         'Ever Married, No Heart Disease', 'Ever Married, Heart Disease'],
        [stroke[(stroke['ever_married'] == 'No') & (stroke['heart_disease'] == 0) & (stroke['stroke'] == 1)].shape[0],
         stroke[(stroke['ever_married'] == 'No') & (stroke['heart_disease'] == 1) & (stroke['stroke'] == 1)].shape[0],
         stroke[(stroke['ever_married'] == 'Yes') & (stroke['heart_disease'] == 0) & (stroke['stroke'] == 1)].shape[0],
         stroke[(stroke['ever_married'] == 'Yes') & (stroke['heart_disease'] == 1) & (stroke['stroke'] == 1)].shape[0]],
        color=['green', 'red', 'purple', 'blue'], alpha=0.5)
plt.ylabel('Count')
plt.title('Ever Married vs Heart Disease Distribution for Stroke Cases')
plt.xticks(rotation=45)
plt.show()
 #%%
stroke_counts = stroke['stroke'].value_counts()
plt.bar(stroke_counts.index, stroke_counts.values)
plt.xticks([0, 1], ['No Stroke', 'Stroke'])
plt.ylabel('Count')
plt.title('Stroke Cases Distribution')
plt.show()

fig = plt.figure(facecolor='white')
stroke_counts = stroke['stroke'].value_counts()
plt.pie(stroke_counts.values, labels=stroke_counts.index, autopct='%1.1f%%')
plt.title('Stroke Cases Distribution')
plt.show()


#%%
corr_matrix = stroke_full_variables.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Plot')
plt.show()
# %%
# Preprocessing
print(stroke.head(5))
stroke = stroke.sample(frac=1, random_state=42).reset_index(drop=True)
print(stroke.head(5))
stroke['bmi'].fillna(value=stroke['bmi'].mean(),inplace=True)
# Perform label encoding for categorical variables
le = LabelEncoder()
stroke['gender'] = le.fit_transform(stroke['gender'])
stroke['ever_married'] = le.fit_transform(stroke['ever_married'])
stroke['work_type'] = le.fit_transform(stroke['work_type'])
stroke['Residence_type'] = le.fit_transform(stroke['Residence_type'])
stroke['smoking_status'] = le.fit_transform(stroke['smoking_status'])
stroke = pd.get_dummies(stroke, columns=['gender','hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type','smoking_status'])
#%%
# Feature Engineering:
stroke = stroke[['age', 'hypertension_0', 'hypertension_1', 'heart_disease_0', 'heart_disease_1', 'stroke']]
X = stroke.drop('stroke', axis=1)
y = stroke['stroke']
y = le.fit_transform(y)


#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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
print(classification_report(y_test, y_pred))
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

y_pred = clf.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Model evaluation
print(classification_report(y_test, y_pred))
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
    'pca__n_components': [1,2,3,4,5],
    # 'mlp__hidden_layer_sizes': [(2,2,2,2), (3,3,3), (2,3,2), (30,),(10,),(5,),(100,)],
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

# Model evaluation
print(classification_report(y_test, y_pred))
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
print(classification_report(y_test, y_pred_keras_binary))
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
print(classification_report(y_test, y_pred_k))
print(f'Accuracy: {accuracy_score(y_test, y_pred_k):.3f}')
print(f'Precision: {precision_score(y_test, y_pred_k):.3f}')
print(f'Recall: {recall_score(y_test, y_pred_k):.3f}')
print(f'F1-score: {f1_score(y_test, y_pred_k):.3f}')
# %%
