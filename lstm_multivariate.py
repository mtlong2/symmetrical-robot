from math import sqrt
from numpy import concatenate, random
from pandas import read_csv, concat, DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import SGD
from matplotlib import pyplot

random.seed(6)
# convert data to a supervised learning problem, (t-1)
def multivariate_to_supervised(data, lag=1, dropnan=True):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    if dropnan:
        df.dropna(inplace=True)
    return df

""" load dataset (from local data folder) ...
For this example, Consumption_kWh is redundant of power_kW*4 and is dropped
GHI is a composite of DHI and DNI, and will be used while dropping the other two.
DHI or DNI may be better choice if the focus is concentrated solar power or concentrated
photovoltaic. """

dataset = read_csv('PowerWeather.csv', header=0, infer_datetime_format=True, index_col=['Date'])
dataset = DataFrame(dataset)

dataset = dataset.drop(columns=['Consumption_kWh','DHI','DNI'], axis=1)
values = dataset.values

#  make all data float
values = values.astype('float64')
# standardize features
scaler = StandardScaler()
scaled = scaler.fit_transform(values)
# frame as supervised learning
supervised = multivariate_to_supervised(scaled, lag=1)
supervised = supervised.iloc[:,:13]

 
# split into train and test sets
values = supervised.values
#train, test = values[:-5000], values[-5000:]
train, test = values[:-10037], values[-10037:]

# split into input and outputs
trainX, train_y = train[:, :-1], train[:, -1]
testX, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
print(trainX.shape, train_y.shape, testX.shape, test_y.shape)
 
# Define network
model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))
optimizer = SGD(lr=0.02, momentum=0.8)
model.add(Dense(1))
model.compile(loss='mse', optimizer=optimizer)
model.summary()

# fit network
history = model.fit(trainX, train_y, epochs=20, batch_size=32, validation_data=(testX, test_y), verbose=1, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()
 
# make a prediction
prediction = model.predict(testX)

# reshape data for inversion
testX = testX.reshape((testX.shape[0], testX.shape[2]))

# invert scaling for forecast
inv_pred = concatenate((prediction, testX[:, 1:]), axis=1)
inv_pred = scaler.inverse_transform(inv_pred)
inv_pred = inv_pred[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_actual = concatenate((test_y, testX[:, 1:]), axis=1)
inv_actual = scaler.inverse_transform(inv_actual)
inv_actual= inv_actual[:,0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_actual, inv_pred))
print('Test RMSE: %.3f' % rmse)

# plot actual v predicted last 1000 point
x = range(1, len(inv_actual)+1)
pyplot.plot(inv_actual, label='original')
pyplot.plot(inv_pred, label='predicted')
pyplot.xticks(x)
pyplot.legend()
pyplot.show()
