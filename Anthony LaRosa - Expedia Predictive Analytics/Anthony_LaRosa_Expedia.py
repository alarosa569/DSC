# Anthony LaRosa
# 1/28/21
# DSC630 - Expedia
# Professor Werner

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from minepy import MINE



# Citation: https://towardsdatascience.com/a-machine-learning-approach-building-a-hotel-recommendation-engine-6812bfd53f50
# A part of this assignment that I found particularly challenging was the feature engineering in how to split and
# extract the year/month data using lambda functions. I remember doing a tutorial on this in DSC550 but I still
# had to look back on TowardsDataScience, and they actually had first run that applied to the assignment. Some of the
# things that they did were actually beyond me even following along, because I don't like using stuff I don't
# understand. For example - I don't understand why they were doing what they did with the pivot table. Maybe we can
# catch up with this on Wednesday.

# Load data into a dataframe
# this is interesting from TDS because I saw a discussion in Slack about not having the space to handle 4gb files
# on disk, and i didn't know you can leave it compressed and sample from it through the PD package

# EDIT 1 - the compression method i mentioned above
#          did not work how i expected it to and took very long, I'm going to unzip the file and try nrows
# EDIT 2 - I tried without nrows, couldn't even read the DF it required almost 6gb of disk
# EDIT 3 - What I mentioned I didn't understand above about the pivot table, i understand now that I did the assignment
#          hands on, but i did not choose to go how the tutorial went and went my own path
# EDIT 4 - Was pushing to learn some new or unfamiliar concepts with this assignment. At first I did the requirements
#          and split the test/train, but then i read about k-fold cross-validation and tried that

hotel_train = "hoteltrain.csv"
hotel_df = pd.read_csv(hotel_train, nrows=300000).dropna()
# I'm going to take a very small sample, 0.01%, 10x less than the example and then turn it up if needed

# verify the shape
print(hotel_df.shape)

# verify the df
pd.set_option('display.max_columns', None)
print(hotel_df)

# perform EDA and look at the distribution of the target variable, the hotel cluster
plt.hist(hotel_df['hotel_cluster'], bins=30)
plt.title("Distribution of Hotel Cluster")
plt.show()

# next is what I mentioned in the beginning that I wasn't sure how to extract the date time but found a tutorial
# utilizing the data time package

def get_year(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').year
        except ValueError:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year
    else:
        return 2013
    pass


def get_month(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').month
        except:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month
    else:
        return 1
    pass


def left_merge_dataset(left_dframe, right_dframe, merge_column):
    return pd.merge(left_dframe, right_dframe, on=merge_column, how='left')


# so now that the functions are defined we are going to have a new column for the year and month that are extracted
# create the new features
hotel_df['date_time_year'] = pd.Series(hotel_df.date_time, index=hotel_df.index)
hotel_df['date_time_month'] = pd.Series(hotel_df.date_time, index=hotel_df.index)

# populate the features with the month and year respectively based on the functions we created above
hotel_df.date_time_year = hotel_df.date_time_year.apply(lambda x: get_year(x))
hotel_df.date_time_month = hotel_df.date_time_month.apply(lambda x: get_month(x))

# remove the original feature
del hotel_df['date_time']

# re-verify the data frame
print(hotel_df)

# this looks good, and im looking forward to using this assignment as reference in the future for handling datetime
# now to do the same thing again for the search columns that need to be parsed properly

hotel_df['srch_ci_year'] = pd.Series(hotel_df.srch_ci, index=hotel_df.index)
hotel_df['srch_ci_month'] = pd.Series(hotel_df.srch_ci, index=hotel_df.index)
# convert year & months to int
hotel_df.srch_ci_year = hotel_df.srch_ci_year.apply(lambda x: get_year(x))
hotel_df.srch_ci_month = hotel_df.srch_ci_month.apply(lambda x: get_month(x))
# remove the srch_ci and srch_co column
del hotel_df['srch_ci']
del hotel_df['srch_co']

# validate feature engineering was successful
print("######################## End DATETIME MODS ########################")
print(hotel_df)
# continuing the EDA looking for a way to find the best feature selection for building the model
print("######################## Below is my Pearson Cor ########################")
print(hotel_df.corr(method='pearson')["hotel_cluster"].sort_values())

# this looks promising but based on the assignment description and my small row sample, I'm going to assume that
# there is no linear cor and investigate methods of finding non-lin feature cor

# initiate a mine algo with default parameters
print("######################## Below is my MIC Analysis ########################")
mine = MINE(alpha=0.6, c=15)
mine.compute_score(hotel_df['srch_destination_id'], hotel_df["hotel_cluster"])
print("The cross-validated Pearsons with MIC for search destination is " + str(mine.mic()))

# since I got this to work on my single use case with interesting results, I wanted to write a loop to see it
# and compare it for all applicable columns
hotel_columns = list(hotel_df)
print(hotel_columns)
# side note, below was the first time i used a for-loop to actually make my life easier and not because it was an
# assignment requirement, and it was great.
for features in hotel_columns:
    mine.compute_score(hotel_df[features], hotel_df["hotel_cluster"])
    print("The MIC for feature " + str(features) + " is " + str(mine.mic()))

# this is very interesting, I've never ran into MIC before and there's not much information out there on it
# compared to other methods. It's supposed to be a cutting edge approach for finding non-linear associations in large
# datasets
#
# For my strategy, I'm going to build my predictor variables based off MIC analysis
# from the output of this analysis, i can build my predictors
# heres an example of something i found fascinating reading about MINE and uncovering 'hidden features'
# when the author in towards data science did there work on this data, they discovered X set of predictors from their
# EDA. When i did it with MIC, i came to the same conclusions + ones that did not appear on the pearsons or their analysis

# locate the records that we know the individual is booking
df = hotel_df.loc[hotel_df['is_booking'] == 1]

# To split the dataset into features and target variables, first create a variable for the feature columns
feature_cols = ['hotel_market', 'hotel_country', 'hotel_continent', 'srch_destination_id', 'orig_destination_distance']
# Set X equal to the feature columns
X = hotel_df[feature_cols]
# Set Y equal to the target variable
y = hotel_df.hotel_cluster
# Step 24.	Using the train_test_split() function, split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# number of samples in each set
print("######################## Below is my Training and Test Data Split ########################")
print("No. of samples in training set: ", X_train.shape[0])
print("No. of samples in validation set:", X_test.shape[0])

print("######################## Below is my Multi-Logistic Regression ########################")
clf = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(multi_class='ovr'))
print(np.mean(cross_val_score(clf, X, y, cv=10)))

print("######################## Below is my SVM ########################")
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(decision_function_shape='ovo'))
print(np.mean(cross_val_score(clf, X, y, cv=10)))
