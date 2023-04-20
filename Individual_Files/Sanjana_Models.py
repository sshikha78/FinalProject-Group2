#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix

#%%
df = pd.read_csv('stroke.csv')

#%%
# Filling Null Data
# We will fill the null values of bmi column with the mean of this column
df['bmi'].fillna(value=df['bmi'].mean(),inplace=True)
#print("After filling null values:",df.isna().sum())

# Drop the ID column as it is not relevant for modeling
#df.drop('id', axis=1, inplace=True)

# Convert the stroke column to integers (0 for no stroke, 1 for stroke)
df['stroke'] = df['stroke'].astype(int)

# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

#%%
X = df.drop(['stroke'], axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
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

#%%
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
# %%
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


# %%
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

# %%
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

# %%
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

# Scale the test data using the same scaler used for training data
X_test_scaled = scaler.transform(X_test)

# Confusion Matrix for KNN
plot_confusion_matrix(knn, X_test_scaled, y_test, cmap='Blues')
plt.title('KNN Confusion Matrix')

# ROC Curve for KNN
plot_roc_curve(knn, X_test_scaled, y_test)
plt.title('KNN ROC Curve')

# %%
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

# Scale the test data using the same scaler used for training data
X_test_scaled = scaler.transform(X_test)

# Confusion Matrix for SVM
plot_confusion_matrix(svm, X_test_scaled, y_test, cmap='Blues')
plt.title('SVM Confusion Matrix')

# ROC Curve for SVM
plot_roc_curve(svm, X_test_scaled, y_test)
plt.title('SVM ROC Curve')

# %%
