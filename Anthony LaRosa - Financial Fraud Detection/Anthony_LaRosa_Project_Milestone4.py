# Anthony LaRosa
# 1/15/21
# DSC630 - Financial Fraud Detection
# Professor Werner

import pandas as pd
import yellowbrick
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from minepy import MINE

# Step 1:  Load data into a dataframe


fraud = "fraud_data.csv"
fraud_df = pd.read_csv(fraud, nrows=50000)

# Step 2:  check the dimension of the table
print("The dimension of the table is: ", fraud_df.shape)
# Step 3:  Look at the data
print(fraud_df.head(5))
# Step 5:  what type of variables are in the table
print("Describe Data")
print(fraud_df.describe())
print("Summarized Data")
print(fraud_df.describe(include=['O']))

# Step 6: display length of data
print('\n')
print("The length of the data is " + str(len(fraud_df)))

# Step 7: display min, max of balances rating
print('\n')
print("Below is the min/max of the old balances")
print(fraud_df['oldbalanceOrg'].min())
print(fraud_df['oldbalanceOrg'].max())
print("Below is the min/max of the new balances")
print(fraud_df['newbalanceOrig'].min())
print(fraud_df['newbalanceOrig'].max())

# Step 8: Create bar charts looking for outliers
ax = fraud_df['type'].value_counts().sort_index().plot(kind='bar', fontsize=12, figsize=(12,10))
ax.set_title('Financial Transaction Type\n', fontsize=18)
ax.set_xlabel('Type', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
plt.show()

fraud_box = fraud_df.boxplot()
plt.show()

# As I continued the EDA in the steps below I realized I need to dummies the 'type' feature
# Convert the categorical Style variable into binary to allow for quantitative analysis
dummyfraud_df = pd.get_dummies(fraud_df, drop_first=True)
# verify the new dataframe
print(dummyfraud_df.head(5))
# feature reduction
# when i dummied the dataframe i exploded into have over 16,000 features in a data set so small there was no value
# i thought about the simpliest way to reduce the features and keep what i needed, and below is what I came up with
dummyfraud_df.drop(list(dummyfraud_df.filter(regex = 'nameOrig')), axis = 1, inplace = True)
dummyfraud_df.drop(list(dummyfraud_df.filter(regex = 'nameDest')), axis = 1, inplace = True)

#  feature selection for building the model
print("######################## Below is my Pearson Cor ########################")
print(dummyfraud_df.corr(method='pearson')["isFraud"].sort_values())

# As i raise the sample size what is becoming clear is the 'amount' feature has a high positive cor to the isFraud feature
# the challenge is that the other features do not seem to be highly cor. I want to try to use the MIC processes like
# i used on the hotel process

print("######################## Below is my MIC Analysis ########################")
mine = MINE(alpha=0.6, c=15)
fraud_columns = list(dummyfraud_df)
print(fraud_columns)
for features in fraud_columns:
    mine.compute_score(dummyfraud_df[features], dummyfraud_df["isFraud"])
    print("The MIC for feature " + str(features) + " is " + str(mine.mic()))

# # Step 21.	To split the dataset into features and target variables, first create a variable for the feature columns
feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
# Set X equal to the feature columns
X = dummyfraud_df[feature_cols]
# Set Y equal to the target variable
y = dummyfraud_df.isFraud
# Using the train_test_split() function, split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# number of samples in each set
print("No. of samples in training set: ", X_train.shape[0])
print("No. of samples in validation set:", X_test.shape[0])
#
# fraud and not fraud breakdown
# print('\n')
print('Number of fraud and non fraud related transactions in training set:')
print(y_train.value_counts())
#
print('\n')
print('Number of fraud and non fraud related transactions in the validation set:')
print(y_test.value_counts())
# Instantiate the model using default parameters
logreg = LogisticRegression()
# Fit the model with data
logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)
# Evaluate the model using a confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
# Heatmap the confusion matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#  create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
# Print the detailed results of the confusion matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
# Create an ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#
# Train Model for Random Forest
print('\n')
print("START OF RANDOM FOREST")
# set up test and training data based of numeric 'Stars' versus Binary 'Stars' used in logistic regression
# # I was testing back and forth here to see which one was more accurate for the forest
# # Set Xn equal to the feature columns
# Xn = dummyramen_df[feature_cols]
# # Step 23.	Set Y equal to the target variable
# yn = dummyramen_df.Stars
# # Step 24.	Using the train_test_split() function, split the data into test and train
# Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.25, random_state=0)
# # number of samples in each set
# print("No. of samples in the Star feature outcome training set: ", Xn_train.shape[0])
# print("No. of samples in Star feature outcome validation set:", Xn_test.shape[0])
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# test the model with 1000 decision trees
# I find it interesting that over 1,000 estimators doesnt improve the metrics
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
#

# train the model on the training data
rf.fit(X_train, y_train)
#

# use the forest predict method on the test data set
predictions = rf.predict(X_test)
print('\n')

# # Calculate the errors
errors = abs(predictions - y_test)

print('The mean absolute error is: ', round(np.mean(errors), 2), 'degrees.')

#  Determine the performance metrics
print("Below are my Random Forest Performance Metrics")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("The accuracy of my Random Forest is Below")
print(accuracy_score(y_test, y_pred))
#
# KERAS NEURAL Network
print("START OF KERAS NEURAL NETWORK")
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras_metrics
#
# define the keras model
# I set input_dim to 41 because that is the number of columns the network will expect in a row
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
#
#  fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=10)
#
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
#
print("########## FINAL RESULTS ##########")
print("Accuracy of Logistic Regression: 97.44% ")
print("Accuracy of Random Forest: 97.44% ")
print("Accuracy of Keras Neural Network: 99.32% ")
print("########## END RESULTS ##########")

