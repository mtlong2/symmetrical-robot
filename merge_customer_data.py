# client customer data - resampling - merging multiple data files
# manipulating different datasets for neural networks
# Power data in 15 min intervals, weather data in 30

from pandas import read_csv, DataFrame, to_datetime, merge, concat
from datetime import datetime

# import dataset (from local folder), resample and align
dataset = read_csv('weather_data.csv', header=2, infer_datetime_format=True)
print(dataset.head(), dataset.dtypes)


# create a column date that includes year, month, hour and minute columns to be upsampled later
dataset['Date'] = to_datetime(dataset[['Year','Month','Day','Hour','Minute']])
dataset = DataFrame(dataset)

# set Date column as index
dataset = dataset.set_index('Date')

# drop the first 5 columns that were Yr,Mon,D,H,Minute
dataset = dataset.drop(dataset.columns[[0,1,2,3,4]], axis=1)
print(dataset.head())

# set all data types to float64 
dataset = dataset.astype('float64')
# confirm correct datatypes
print(dataset.dtypes)

# resample to 15 min intervals
dataset15 = dataset.resample('15min').interpolate()
print(dataset15.head())

# export 15 min to csv for future use
dataset15.to_csv('weather_data15min.csv')

# load 15 min weather and 15 power datasets together.  
def parse(x):
	return datetime.strptime(x, '%m/%d/%y %H:%M')
left = read_csv('power_15min.csv', header=0, infer_datetime_format=True, index_col=0, date_parser=parse)
right = read_csv('weather_data15min.csv', header=0, infer_datetime_format=True, index_col=0)

#  weather data is limited to 1 year, reduce power data to same size
left = left[:-98]
right = right.drop(right.index[[0,1]], axis=0)

left, right = DataFrame(left), DataFrame(right)
print(left.head(), right.head(), left.describe(), right.describe())

# concatenate the two dataframes together
frames = [left, right]

PowerWeather = concat(frames, axis=1, sort=True)
print(PowerWeather.head())

PowerWeather.to_csv('PowerWeather.csv')

