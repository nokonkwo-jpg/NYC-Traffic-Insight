import pandas as pd

weather_files = ['RawDataFiles/weather_data_vm1.csv', 'RawDataFiles/weather_data_vm2.csv',
'RawDataFiles/weather_data_vm3.csv', 'RawDataFiles/weather_data_vm4.csv',
'RawDataFiles/weather_data_vm5.csv','RawDataFiles/weather_data_vm6.csv', 'RawDataFiles/weather_data_vm7.csv']

wFrames = [pd.read_csv(file) for file in weather_files] #reads each file in weather files

finalWeatherframe = pd.concat(wFrames, ignore_index=True)

#Understanding the data
print(finalWeatherframe.columns)
print(finalWeatherframe.describe())
print(finalWeatherframe.shape)
print(finalWeatherframe.dtypes)
print(finalWeatherframe.tail())
print(len(finalWeatherframe))

print(finalWeatherframe.isnull().sum())
print(finalWeatherframe.isna().any())
#found null @ columns uv_index(915360), freezing_level_height(915360), visibility(915360), snow_depth(91536)
finalWeatherframe = finalWeatherframe.drop(['uv_index', 'freezing_level_height', 'visibility'], axis='columns')
print(finalWeatherframe.columns)

#cleaning formatting of the date-time
finalWeatherframe['date'] = pd.to_datetime(finalWeatherframe['date'])
finalWeatherframe['date'] = finalWeatherframe['date'].dt.strftime('%Y-%m-%d %H:%M')
#Now in format (year-month-day hour:minute)
print(finalWeatherframe['date'].head(10))

#find dupes (0 found! 😎)
print(finalWeatherframe.duplicated().sum())
print(finalWeatherframe.dtypes)

#changing borough datatype to be category
finalWeatherframe['borough'] = finalWeatherframe['borough'].astype('category')