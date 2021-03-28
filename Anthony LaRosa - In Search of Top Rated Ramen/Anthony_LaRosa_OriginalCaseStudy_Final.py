# Anthony LaRosa
# 10/8/20
# DSC550 Case Study Part 1
# Professor Werner

import pandas as pd
import yellowbrick
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Step 1:  Load data into a dataframe


ramen = "ramen-ratings.csv"
ramen_df = pd.read_csv(ramen)

# Step 2:  check the dimension of the table
print("The dimension of the table is: ", ramen_df.shape)
# Step 3:  Look at the data
print(ramen_df.head(5))
# Step 5:  what type of variables are in the table
print("Describe Data")
print(ramen_df.describe())
print("Summarized Data")
print(ramen_df.describe(include=['O']))

# Step 6: display length of data
print('\n')
print("The length of the data is " + str(len(ramen_df)))

# Step 7: display min, max of ramen rating
print('\n')
print("Below is the min/max of the ramen ratings")
print(ramen_df['Stars'].min())
print(ramen_df['Stars'].max())

# Step 8: Remove any row that the stars variable is equal to unrated
ramen_df = ramen_df[ramen_df.Stars != 'Unrated']
# verify min/max after cleaning
print("Below is the min/max of the ramen ratings after cleaning")
print(ramen_df['Stars'].min())
print(ramen_df['Stars'].max())

# Step 9: Create bar charts of country, brand , and style to look for outliers
ax = ramen_df['Country'].value_counts().sort_index().plot(kind='bar', fontsize=12, figsize=(12,10))
ax.set_title('Country of Origin\n', fontsize=18)
ax.set_xlabel('Country', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
# plt.show()

# ax = ramen_df['Brand'].value_counts().sort_index().plot(kind='bar', fontsize=12, figsize=(12,10))
# ax.set_title('Maker of Ramen\n', fontsize=18)
# ax.set_xlabel('Brand', fontsize=16)
# ax.set_ylabel('Count', fontsize=16)
# plt.show()

# there are too many brands for a box plot

ax = ramen_df['Style'].value_counts().sort_index().plot(kind='bar', fontsize=12, figsize=(12,10))
ax.set_title('Style of Ramen\n', fontsize=18)
ax.set_xlabel('Style', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
# plt.show()

# Step 10: from this, I found there is a double of USA and United States labels when there should be only one
# there's very few from 'united states' labels, I am considering it an outlier and removing it
ramen_df = ramen_df[ramen_df.Country != 'United States']
# verification of cleaning
ax = ramen_df['Country'].value_counts().sort_index().plot(kind='bar', fontsize=12, figsize=(12,10))
ax.set_title('Country of Origin\n', fontsize=18)
ax.set_xlabel('Country', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
# plt.show()

# Step 11 Convert the stars variable to a float so that it can be sorted properly
ramen_df['Stars'] = ramen_df['Stars'].astype(float)

print("####################### Part 2 #######################")

# Step 12 Based on the summarized data review, remove the variety feature
print(ramen_df.head(5))
ramen_df = ramen_df.drop(columns=['Variety'])
print(ramen_df.head(5))

# Step 13  Remove the brand
ramen_df = ramen_df.drop(columns=['Brand'], axis=1)

# Step 14 Verify the dataframe with the features dropped
print(ramen_df.head(5))

# Step 15 Generate and review the average rating by country
print(ramen_df.groupby('Country', as_index=False)['Stars'].mean())

# Step 16 Generate and review the average rating by style
print(ramen_df.groupby('Style', as_index=False)['Stars'].mean())

# In step 16 I found another outlier to clean up. A single review which was done at a "bar"
# not only that, it was given a perfect 5.0.

# searched for the "bar" entry
# Step 17.	Find and remove row index 1425 which has outlier ‘bar’ and ‘5.0’
print(ramen_df.loc[[1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429]])
ramen_df = ramen_df[ramen_df.Style != 'Bar']

# Verify that bar was dropped
print(ramen_df.groupby('Style', as_index=False)['Stars'].mean())

print(ramen_df)

# # Step 19 Convert the categorical Style variable into binary to allow for quantitative analysis
dummyramen_df = pd.get_dummies(ramen_df, drop_first=True)
# # verify the new dataframe
print(dummyramen_df.head(5))
# Step 20 remove the review number column
dummyramen_df = dummyramen_df.drop(columns=['Review_Number'], axis=1)
print(dummyramen_df)
print("####################### Part 2 #######################")

print("####################### Part 3 #######################")

# iterating the columns
for col in dummyramen_df.columns:
    print(col)

# Step 21.	To split the dataset into features and target variables, first create a variable for the feature columns
feature_cols = ['Style_Box',
'Style_Can',
'Style_Cup',
'Style_Pack',
'Style_Tray',
'Country_Bangladesh',
'Country_Brazil',
'Country_Cambodia',
'Country_Canada',
'Country_China',
'Country_Colombia',
'Country_Dubai',
'Country_Estonia',
'Country_Fiji',
'Country_Finland',
'Country_Germany',
'Country_Ghana',
'Country_Holland',
'Country_Hong Kong',
'Country_Hungary',
'Country_India',
'Country_Indonesia',
'Country_Japan',
'Country_Malaysia',
'Country_Mexico',
'Country_Myanmar',
'Country_Nepal',
'Country_Netherlands',
'Country_Nigeria',
'Country_Pakistan',
'Country_Philippines',
'Country_Poland',
'Country_Sarawak',
'Country_Singapore',
'Country_South Korea',
'Country_Sweden',
'Country_Taiwan',
'Country_Thailand',
'Country_UK',
'Country_USA',
'Country_Vietnam']
# Step 22.	Set X equal to the feature columns
X = dummyramen_df[feature_cols]
# Step 23.	Set Y equal to the target variable
y = dummyramen_df.Top_Rated
# Step 24.	Using the train_test_split() function, split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# number of samples in each set
print("No. of samples in training set: ", X_train.shape[0])
print("No. of samples in validation set:", X_test.shape[0])

# Survived and not-survived
print('\n')
print('Number of top rated and not top rated training set:')
print(y_train.value_counts())

print('\n')
print('Number of top rated and not top rated in the validation set:')
print(y_test.value_counts())
# Step 25.	Instantiate the model using default parameters
logreg = LogisticRegression()
# Step 26.	Fit the model with data
logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)
# Step 27.	Evaluate the model using a confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
# Step 28.	Heatmap the confusion matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# plt.show()
# Step 29.	Print the detailed results of the confusion matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
# Step 30.	Create an ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
#plt.show()

# Train Model for Random Forest
print('\n')
print("START OF RANDOM FOREST")
# set up test and training data based of numeric 'Stars' versus Binary 'Stars' used in logistic regression
# I was testing back and forth here to see which one was more accurate for the forest
# Set Xn equal to the feature columns
Xn = dummyramen_df[feature_cols]
# Step 23.	Set Y equal to the target variable
yn = dummyramen_df.Stars
# Step 24.	Using the train_test_split() function, split the data into test and train
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.25, random_state=0)
# number of samples in each set
print("No. of samples in the Star feature outcome training set: ", Xn_train.shape[0])
print("No. of samples in Star feature outcome validation set:", Xn_test.shape[0])
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# Step 31
# test the model with 1000 decision trees
# I tested moving this from 1000, 2000, and 9000 estimators and no notable increase in performance
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Step 32
# train the model on the training data
rf.fit(X_train, y_train)

# Step 33
# use the forest predict method on the test data set
predictions = rf.predict(X_test)
print('\n')
# print("BELOW ARE MY RANDOM FORREST PREDICTIONS")
# print(predictions)
# Step 34
# Calculate the errors
errors = abs(predictions - y_test)
# print('\n')
# print("BELOW ARE MY ERRORS")
# print(errors)
# print('y-test')
# print(y_test)

# I commented out alot of prints, they were all used for troubleshooting and validation

# Step 35
# print the  mean absolute error
print('The mean absolute error is: ', round(np.mean(errors), 2), 'degrees.')

# Determine the performance metrics
# Step 36
print("Below are my Random Forrest Performance Metrics")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# KERAS NEURAL Network
print("START OF KERAS NEURAL NETWORK")
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Step 37 define the keras model
# I set input_dim to 41 because that is the number of columns the network will expect in a row
model = Sequential()
model.add(Dense(12, input_dim=41, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 38 compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 39 fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Step 40 evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

print("########## FINAL RESULTS ##########")
print("Accuracy of Logistic Regression: 63% ")
print("Accuracy of Random Forrest: 67% ")
print("Accuracy of Keras Neural Network: 64% ")
print("########## END RESULTS ##########")
