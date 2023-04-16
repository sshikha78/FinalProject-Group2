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
# Perform label encoding for categorical variables
le = LabelEncoder()
stroke['gender'] = le.fit_transform(stroke['gender'])
stroke['ever_married'] = le.fit_transform(stroke['ever_married'])
stroke['work_type'] = le.fit_transform(stroke['work_type'])
stroke['Residence_type'] = le.fit_transform(stroke['Residence_type'])
stroke['smoking_status'] = le.fit_transform(stroke['smoking_status'])


X = stroke.drop('stroke', axis=1)
y = stroke['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%


