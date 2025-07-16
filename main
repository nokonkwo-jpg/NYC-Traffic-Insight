import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

traffic = pd.read_csv("RawDataFiles/TrafficData.csv")
weather = pd.read_csv("RawDataFiles/finalWeather.csv")

traffic.drop_duplicates(inplace=True)
traffic.drop(columns= ['MM'], inplace=True)
traffic["WktGeom"] = traffic["WktGeom"].str.strip("POINT ()")
traffic.drop(columns= ['SegmentID'], inplace=True)
traffic[["Latitude", "Longitude"]] = traffic['WktGeom'].str.split(' ', n=2, expand=True)
traffic = traffic.drop(['WktGeom'], axis=1)

traffic.fillna('', inplace=True)

#print(weather.dtypes)
#print(traffic.dtypes)

#Prep data for merge
weather = weather.rename(columns={'borough':'Boro'})
traffic['Boro'] = traffic['Boro'].astype('category')
weather['Boro'] = weather['Boro'].astype('category')
print("weather columns:", weather.columns.tolist())
print("traffic columns:", traffic.columns.tolist())

print(weather.dtypes)
print(traffic.dtypes)

weather['date'] = pd.to_datetime(weather['date'])
weather['Yr'] = weather['date'].dt.year
weather['M'] = weather['date'].dt.month
weather['D'] = weather['date'].dt.day
weather['HH'] = weather['date'].dt.hour
merged_traffic_data = pd.merge(traffic, weather, on=['Yr', 'M', 'D', 'HH', 'Boro'])
merged_traffic_data.to_csv("RawDataFiles/merged_traffic_data.csv")
