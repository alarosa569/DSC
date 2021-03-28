# Course: DSC510-T302
# Assignment: 12.1
# Date: 05/15/2020
# Name: Anthony LaRosa
# Description: Final Project: Weather Program

import json
import string
import requests


def city_work(city_name):  # function for city name lookup
    api = "http://api.openweathermap.org/data/2.5/weather?q={0},US&appid={1}&units=imperial".format(city_name,
                                                                                                    "25b17cab1cde9a0a3e2f649cee1a7d5f")
    # set the api for connecting
    # print(api)  # DEBUG
    try:
        weather_response = requests.request("GET", api)  # perform the get request to the API
    except EXCEPTION as err:
        print("Failed to retrieve ", err)  # print the error exception if it fails to retrieve
    if weather_response.status_code == 200:  # do a status code check and if not 200 then go to else statement
        print("Successfully connected to the weather server")
        print("Your weather for {0} can be seen below".format(city_name))
        weather_data = weather_response.text
        try:
            weather_info = json.loads(weather_data)  # store the text and load it into json so we can get key/vals
        except EXCEPTION as parse_err:
            print("Unable to load JSON due to ", parse_err)  # validation if JSON fails to load
        pretty_print(weather_info)  # send the output of joke_info to the pretty print function for formatting
        print("How about we check the weather in another city?")
    else:
        print("Could not connect to the LaRosa View weather service")


def zip_work(zip_code):  # function for zip code lookup
    api = "http://api.openweathermap.org/data/2.5/weather?zip={0},US&appid={1}&units=imperial".format(zip_code,
                                                                                                      "25b17cab1cde9a0a3e2f649cee1a7d5f")
    #  set the API for connecting
    #  print(api)  # DEBUG
    try:
        weather_response = requests.request("GET", api)  # perform the get request to the API
    except EXCEPTION as err:
        print("Failed to retrieve ", err)  # if the request to the API fails to retrieve notify user
    if weather_response.status_code == 200:  # do a status code check and if not 200 then go to else statement
        print("Successfully connected to the weather server")
        print("Your weather for zip code {0} can be seen below".format(zip_code))
        weather_data = weather_response.text
        try:
            weather_info = json.loads(weather_data)  # store the text and load it into json so we can get key/vals
        except EXCEPTION as parse_err:
            print("Unable to load JSON due to ", parse_err)  # notification if JSON data load fails
        pretty_print(weather_info)  # send the output of joke_info to the pretty print function for formatting
        print("How about we check the weather in another city?")
    else:
        print("Could not connect to the LaRosa View weather service")


def pretty_print(weather_info):  # function for printing a clean output
    weather_main = weather_info.get('main')  # get the dict from main
    print('\n')  # spacing for readability
    print("Current Temperature : " + str(weather_main.get('temp')) + " degrees")  # format and print each field for user
    print("Feels Like : " + str(weather_main.get('feels_like')) + " degrees")
    print("Low Temperature : " + str(weather_main.get('temp_min')) + " degrees")
    print("High Temperature : " + str(weather_main.get('temp_max')) + " degrees")
    print("Pressure : " + str(weather_main.get('pressure')) + " hPa")
    print("Humidity : " + str(weather_main.get('humidity')) + " percent")
    print("Sky Status : " + str(weather_info['weather'][0]['description']))
    print('\n')


def main():
    print("Welcome to LaRosa View! This program will provide you weather information")  # user greeting
    while True:  # While loop to allow the user to lookup multiple entries
        weather_start = input("If you would like to search by City Name press '1' or by zip code, press '2'. When "
                              "finished with the program, press '3' to exit: ")
        if weather_start == "1":
            city_name = input("Enter the name of the city you would like to view the weather for: ")
            if city_name.isalpha():  # check to ensure only alphabetical characters were entered
                city_work(city_name)
            else:  # user notification if invalid input is detected
                print('\n')
                print('*' * 20 + 'ERROR: Invalid Input Detected' + '*' * 20)
                print(str(city_name) + " is not a valid entry. Please enter a valid city name")
                print('*' * 69)
                print('\n')
        elif weather_start == "2":  # zip code initiation and validation on numerals and length
            zip_code = input("Enter the zip code you would like to view the weather for: ")
            if zip_code.isdigit() and len(zip_code) == 5:
                zip_work(zip_code)
            else:
                print('\n')
                print('*' * 20 + 'ERROR: Invalid Input Detected' + '*' * 20)
                print(str(zip_code) + " is not a valid entry. Please enter a valid zip code")
                print('*' * 69)
                print('\n')
        elif weather_start == "3":
            break
        else:
            print("Invalid Input Detected. Must enter '1' , '2' , or '3' ")  # validation check of invalid category
    print("Thank you for using LaRosa View, Goodbye!")  # goodbye message to user


if __name__ == "__main__":
    main()
