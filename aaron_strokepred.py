#%%
# package install
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
# stroke['gender'] = le.fit_transform(stroke['gender'])
# stroke['ever_married'] = le.fit_transform(stroke['ever_married'])
# stroke['work_type'] = le.fit_transform(stroke['work_type'])
# stroke['Residence_type'] = le.fit_transform(stroke['Residence_type'])
# stroke['smoking_status'] = le.fit_transform(stroke['smoking_status'])
stroke = pd.get_dummies(stroke, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
X = stroke.drop('stroke', axis=1)
X = stroke.drop('bmi', axis=1)
y = stroke['stroke']
y = le.fit_transform(y)
#%%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# #%%
# mlp_gs = MLPClassifier(max_iter=200)
# parameter_space = {
#     'hidden_layer_sizes': [(10,30,10),(20,),(10,10),(5,)],
#     'activation': ['tanh', 'relu', 'logistic'],
#     'solver': ['sgd', 'adam', 'lbfgs'],
#     'alpha': [0.0001, 0.05, 0.01, 0.001],
#     'learning_rate': ['constant','adaptive'],
# }
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
# clf.fit(X_train_scaled, y_train) # X is train samples and y is the corresponding labels
#%%
print('Best parameters found:\n', clf.best_params_)
# %%
# Neural Network
from sklearn.neural_network import MLPClassifier
# Neural network model
# mlp = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000)
# mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='tanh',learning_rate='adaptive',max_iter=100)
mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(2,2),(10,),(20,),(10,10),(5,5)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.01, 0.001, 0.000001, 0.5],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train_scaled, y_train) # X is train samples and y is the corresponding labels
print('Best parameters found:\n', clf.best_params_)
# mlp.fit(X_train_scaled, y_train)
# plt.plot(mlp.loss_curve_)
# plt.plot(clf.loss_curve_)
# plt.show()
# %%
# y_pred = mlp.predict(X_test_scaled)
y_pred = clf.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))
#%%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Calculate the false positive rate, true positive rate and thresholds
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

# %%
