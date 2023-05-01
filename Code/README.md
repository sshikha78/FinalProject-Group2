# Stroke-Prediction Code

The file has been created from each group working seperately on their own portion and then merging all working code into this single main file to be run in order with comments of what is taking place for the code.

## Order of Code
### Import Packages
All packages needed for eda, models, and evaluation are at the beginning of the file.
### Read in Data
Simple pandas dataframe reading in the stroke csv file so we may manipulate and train the data later on.
### EDA
Each member of the group has posted their own individual EDA of the variables they were given to the demonstrate distribution and insights.
### Data Preparation
Handling nulls and outliers that were found in the code.
### Pre-Processing 
Scaling quantitative dataset variables and using both get dummies and label encoding to categorical variables.
### Feature deduction 
Reducing the variables that are not correlated to stroke to improve testing later on. 
### Train Split for Models
Developing out train and test data by splitting the dataset into X and Y sets.
### SMOTE
We have sever imbalancing issues and utilized SMOTE to over sample the minority class so stroke target will be even.
### Model Building / Model Evaluation
XGBoost, Gradient Boosting, Random Forest, MLP Classifier, Keras, Logistic Regression, SVM, and K-NN. Each with their own metrics and plot to adequately assess the model's performance. 

