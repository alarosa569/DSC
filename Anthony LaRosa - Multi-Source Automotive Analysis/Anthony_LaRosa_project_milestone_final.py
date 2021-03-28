# Anthony LaRosa
# 11/15/20
# DSC540-Wk11/12
# Professor Williams

# import the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import json
import urllib.request, urllib.parse, urllib.error
from sqlalchemy import create_engine
import sqlite3
import seaborn as sns


# The first thing I did, was combine my last (6) weeks of project work



#Below are the (5) techniques I chose to complete the assignment
"""
Replace Headers 
Format data into a more readable format
Identify outliers and bad data
Find duplicates
Fix casing or inconsistent values
"""

# first I'm going to read my flat file for the auto data into a dataframe
autoflat_df = pd.read_csv("AutoDataFlat.csv", nrows=50, low_memory=False)
# verify the new dataframe
print(autoflat_df.head())

# Find duplicates
# I expect, and there should be duplicates in Make, Model, and EngineIndex (because same engine is used in multiple models)
# But I don't want duplicates in my VehicleID field which should be unique
# use the duplicated function which returns a boolean value based on whether there is duplicates or not
# check the variables Make, Model, Engine Index
print("Make is duplicated - {}".format(any(autoflat_df.Make.duplicated())))
print("Model is duplicated - {}".format(any(autoflat_df.Model.duplicated())))
print("Engine Index is duplicated - {}".format(any(autoflat_df.EngineIndex.duplicated())))
print("VehicleID is duplicated - {}".format(any(autoflat_df.VehicleID.duplicated())))
# this looks good because my make, model, engine is duplicated but not my vehicle ID

# Header transforms
# There's white space in my headers which is posing an issue for me. I'm going to strip out the white space
# and reassign it to my df variable
autoflat_df.columns = autoflat_df.columns.str.replace(')', '')
autoflat_df.columns = autoflat_df.columns.str.replace('(', '')
autoflat_df.rename(columns=lambda x: x.strip(), inplace=True)
autoflat_df = autoflat_df.rename(columns=lambda x: x.strip())
print(autoflat_df.head())

# next I want to convert my headers to all lowercase, when analyzing and manipulating data i like for the
# case to be consistent

autoflat_df.columns = map(str.lower, autoflat_df.columns)
print(autoflat_df.head())

# my style with variables is to have an underscore between multiple word variables
# since I already stripped the whitespace at the ends, I'm now going to replace the whitespace in between
# words with an underscore

autoflat_df.columns = autoflat_df.columns.str.replace(' ', '_')
print(autoflat_df.head())

# now that my headers are in the formats and cleaned how I want them, I want to look at the values in the dataset
# im going to iterate through the columns and produce value counts, if there's columns that are mostly 0's, then i want
# to delete the column
print(autoflat_df.columns)
column_list = autoflat_df.columns
for x in column_list:
    print(autoflat_df[x].value_counts().nlargest(1))

autoflat_df = autoflat_df.drop(columns=['transmission_descriptor', 'city_mpg_ft1', 'unrounded_city_mpg_ft1', 'city_mpg_ft2',
                          'unrounded_city_mpg_ft2', 'city_gasoline_consumption_cd', 'city_electricity_consumption',
                          'highway_mpg_ft2', 'unrounded_highway_mpg_ft2', 'highway_gasoline_consumption_cd',
                          'highway_electricity_consumption', 'unadjusted_highway_mpg_ft1', 'unadjusted_city_mpg_ft2',
                          'unadjusted_highway_mpg_ft2', 'combined_mpg_ft1', 'combined_mpg_ft2', 'unrounded_combined_mpg_ft2',
                          'combined_electricity_consumption', 'combined_gasoline_consumption_cd', 'annual_fuel_cost_ft1',
                          'annual_consumption_in_barrels_ft1', 'tailpipe_co2_ft2', 'my_mpg_data', '2d_passenger_volume', '2d_luggage_volume', '4d_passenger_volume', '4d_luggage_volume', 'hatchback_passenger_volume', 'alternate_charger',
                          'hours_to_charge_120v', 'hours_to_charge_240v', 'hours_to_charge_ac_240v', 'composite_city_mpg', 'composite_highway_mpg', 'composite_combined_mpg', 'range_ft1', 'city_range_ft1', 'range_ft2',
                          'city_range_ft2'])

# verify the column drops
print(autoflat_df.columns)
print(autoflat_df.head())
# this cut my columns down by about half which was part of my project plan

# display the columns after the cleaning
print(autoflat_df.columns)

# removing outliers
# store the original size of the df
og_size = autoflat_df.shape
autoflat_df = autoflat_df[np.isfinite(autoflat_df['unadjusted_city_mpg_ft1'])]
post_size = autoflat_df.shape
# numpys isfinite function checks to make sure it has a non NaN value
# # printing the size difference below

print("The size of the original data was - {prev[0]} rows and the size of the new data is - {after[0]} rows".format(prev=og_size, after=post_size))

# next I will use a boxplot to visualize any outliers for city and highway mpg
plt.boxplot(autoflat_df.unadjusted_city_mpg_ft1, notch=True)
# plt.show()
# im going to pull any mpg over 75 out of the data

plt.boxplot(autoflat_df.unrounded_highway_mpg_ft1, notch=True)
# plt.show()
# same with this one, going to remove over 75mpg.

# # finally we will get rid of outliers. In this case it is removing values over 75 mpg.
trimmedautoflat_df = autoflat_df[(autoflat_df['unadjusted_city_mpg_ft1'] <= 75)]
trimmedautoflat_df = autoflat_df[(autoflat_df['unrounded_highway_mpg_ft1'] <= 75)]
print(trimmedautoflat_df.head())
print("After trimming outliers the new shape of the dataframe is " + str(trimmedautoflat_df.shape[0]))

# set the url to the url for my automobile data which I defined in the first milestone
url = 'https://en.wikipedia.org/wiki/List_of_automobile_sales_by_model'

# open the url
html = urlopen(url)

# utilize beautiful soup from our textbook for the parsing
soup = BeautifulSoup(html, 'html.parser')

# use the find all function to search for the automobile tables

tables = soup.find_all('table')

# I'm getting that my second cell index is out of range in the line 46 onward block so added this to start tshooting
# print(tables)

# next step is, I am going to look for TR tag which is the html tables and then
# the td tags which is the table data. I have four features i want to include which are
# the columns for Production, Sales, and Assembly. They will be strings.

# Create array to hold the data we extract
productions = []
sales = []
assemblies = []

for table in tables:
    rows = table.find_all('tr')

    for row in rows:
        cells = row.find_all('td')

        # I am using len ==4 because when looking at the HTML the only tables im interested in scraping their data have
        # a cell length of 4
        # print(len(cells))
        if len(cells) == 4:
            production = cells[1]
            productions.append(production.text.strip())

            sale = cells[2]
            sales.append(sale.text.strip())

            assembly = cells[3]
            assemblies.append(assembly.text.strip())

# verification of my arrays
print("Below are the production years")
print(productions)

print("Below are the sales")
print(sales)

print("Below are the assembly countries")
print(assemblies)

# create the dataframe
wiki_df = pd.DataFrame(productions, columns=['Production Years'])

# add my two additional columns
wiki_df['Sales Numbers'] = sales
wiki_df['Assembly Country'] = assemblies

# arrange columns in alphabetical order
wiki_df = wiki_df.reindex(sorted(wiki_df.columns), axis=1)
print(wiki_df)

# remove the extraneous non-numeric data from the sales numbers
wiki_df['Sales Numbers'] = wiki_df['Sales Numbers'].str.extract('(\d+)', expand=False)
print(wiki_df)

# remove any blank rows
wiki_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# verify blank rows dropped
print(wiki_df)

# remove any duplicates
wiki_df = wiki_df.drop_duplicates()

# verify blank rows dropped
print(wiki_df)

# clean up yugoslavia value
wiki_df['Assembly Country'].replace(['YugoslaviaSerbia'], 'Yugoslavia', inplace=True)

# verify value update
print(wiki_df)

# set the service URL for Honda dataframe
serviceurl_honda = 'https://vpic.nhtsa.dot.gov/api/vehicles/getmanufacturerdetails/honda?format=json'

# setup the get to the service url
response_honda = requests.get(serviceurl_honda)
# validate the response code
print("Below is my API status code for Honda")
print(response_honda.status_code)
# looks good we get a 200
# now let's check the response for Honda
print(" My validation of Honda API JSON Response is Below")
print(response_honda.json())

# now let's convert the JSON string to a python object
honda_entry = urllib.request.urlopen(serviceurl_honda) # this is opening our fully built url and then reading the data below
honda_data = honda_entry.read()
honda_final = json.loads(honda_data) # loading the JSON as a python object
# now let's check out the final load for honda
print("My validation for the loaded Python Object is below")
print(honda_final)

# next let's load the json python object into a pandas df
honda_df = pd.read_json(honda_data)
# I need to flatten out the results column, so i used the normalized function
hondanormal_df = pd.json_normalize(honda_df['Results'])
print(hondanormal_df)
pd.set_option('display.max_columns', None) # I want to see all the columns for printing
print(hondanormal_df)

# I have to say, I'm kind of surprised with some of the 'dirtiness' of this data. Such as have states listed
# by their shortname and full name. I thought since it was a govt db that it would be more normalized already

# Replace headers
hondanormal_df.rename(columns={'SubmittedName': 'Name', 'SubmittedOn': 'On', 'SubmittedPosition': 'Position'}, inplace=True)
print(hondanormal_df)  # verify updates
# looks good

# clean up California value
hondanormal_df['StateProvince'].replace(['CA'], 'California', inplace=True)

# clean up Indiana value
hondanormal_df['StateProvince'].replace(['IN'], 'Indiana', inplace=True)

# clean up Alabama value
hondanormal_df['StateProvince'].replace(['AL'], 'Alabama', inplace=True)

print(hondanormal_df)  # verify updates
# looks good

# Remove the numeric data from the StateProvince
hondanormal_df.StateProvince = hondanormal_df.StateProvince.str.replace('[0-9]','')
print(hondanormal_df)  # verify updates
# looks good

# Remove rows where the Automaker common name is none, where is going to be used as the common value for the project
hondanormal_df = hondanormal_df.dropna(axis=0, subset=['Mfr_CommonName'])
print(hondanormal_df)  # verify updates
# looks good

# drop any duplicate automaker IDs
hondanormal_df.drop_duplicates(subset='Mfr_ID', keep="last")
print(hondanormal_df)  # verify updates


# ======================= Below I am going to print my (3) cleaned DFs to verify before storing =======================
print("Below is my flat source verification")
print('\n')
print(trimmedautoflat_df)

print("Below is my web source verification")
print('\n')
print(wiki_df)

print("Below is my API source verification")
print('\n')
print(hondanormal_df)

# create the connection for the sqlite database

engine = create_engine('sqlite:///finalproject_540.db', echo=True)
sqlite_connection = engine.connect()

# Create my first table from the flat df
sqlite_table1 = "FlatTable"
trimmedautoflat_df.to_sql(sqlite_table1, sqlite_connection, if_exists='replace')

# Create my second table from the web df
sqlite_table2 = "WebTable"
wiki_df.to_sql(sqlite_table2, sqlite_connection, if_exists='replace')

# First two went smoothly, and getting an error on the API one. I think it's because on of the columns isn't a str dtype
hondanormal_df = hondanormal_df.applymap(str)
# Yep that was it!

# Create my third table from the api df
sqlite_table3 = "APITable"
hondanormal_df.to_sql(sqlite_table3, sqlite_connection, if_exists='replace')

# sqlite_connection.close()

# now I'm going to reconnect to the database and work on merging the the individual tables
# connect to the database
conn = sqlite3.connect('finalproject_540.db')

# create a cursor object
cursor = conn.cursor()

# In this next block of code I wanted to see the base API table, and then with each join how it grew
sql0 = '''SELECT * from APITable'''
cursor.execute(sql0)
result = cursor.fetchall()
print("Total rows in APITable are:  ", len(result))
# set sql statement for first join
# Below are some of my mistaken experiments of thinking that the APITable data gets committed automatically
sql1 = '''SELECT * from APITable CROSS JOIN WebTable CROSS JOIN FlatTable'''
# sql2 = '''SELECT * from APITable CROSS JOIN FlatTable'''
# sql2 = "SELECT * from WebTable"
# sql3 = "SELECT * from FlatTable"

# execute the first join of the two tables
cursor.execute(sql1)
result = cursor.fetchall()
print("Total rows in APITable are:  ", len(result))
# cursor.execute(sql2)
# result = cursor.fetchall()
# print("Total rows in APITable are:  ", len(result))
# print(result)
# cursor.execute(sql3)
conn.commit()

# extract the merged tables into a dataframe for visualizations
mergedtables_df = pd.read_sql_query(sql1, conn)
print(mergedtables_df)
conn.close()

print("Below is a list of the columns in my new merged table dataframe for verification")
print(list(mergedtables_df.columns))

# I used cross joins to combine the tables and extract them into a single dataframe, I will not make me plots of the
# dataframe
ax = mergedtables_df['highway_mpg_ft1'].value_counts().sort_index().plot(kind='bar', fontsize=12, figsize=(12,10))
ax.set_title('All Automobiles by Highway MPG\n', fontsize=18)
ax.set_xlabel('MPG', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
plt.show()

mergedtables_df.plot(x ='engine_cylinders', y='highway_mpg_ft1', kind = 'scatter')
plt.title("Engine Cylinders to Highway MPG")
plt.show()

mergedtables_df.plot(x ='engine_cylinders', y='make', kind = 'scatter')
plt.title("Engine Cylinders to Make")
plt.show()

mergedtables_df.plot(x ='make', y='tailpipe_co2_in_grams/mile_ft1', kind = 'scatter')
plt.title("Make to CO2 Pollution")
plt.show()

mergedtables_df.plot(x ='tailpipe_co2_in_grams/mile_ft1', y='Assembly Country', kind = 'scatter')
plt.title("CO2 Pollution by Assembly Country")
plt.show()

mergedtables_df.plot(x ='make', y='save_or_spend_5_year', kind = 'scatter')
plt.title("Make to Save or Spend")
plt.show()
