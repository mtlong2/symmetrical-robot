""" create three datasets to explore different value scaling options; normalization -1 to 1, 0 to 1 and Z-score -1. to 1"""

from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
from numpy import split, array, arange
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,  LSTM


# convert data to a supervised learning problem, (t-1)
def multivariate_to_supervised(data, lag=1, dropnan=True):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    if dropnan:
        df.dropna(inplace=True)
    return df

# import dataset (from local folder) and scale into 3 values
dataset = read_csv('PowerWeather.csv', header=0, infer_datetime_format=True, index_col=['Date'])

"""# For this example, Consumption_kWh is redundant of power_kW*4 and is dropped
GHI is a composite of DHI and DNI, and will be used while dropping the other two.
DHI or DNI may be better choice if the focus is concentrated solar power or concentrated
photovoltaic. """

dataset = DataFrame(dataset)
dataset = dataset.drop(columns=['Consumption_kWh','DHI','DNI'], axis=1)
print(dataset.head())
dataset = dataset.values
dataset = dataset.astype('float64')
#normalize data b/w 0 and 1 and -1 to 1
scaler0to1, scaler1to1 = MinMaxScaler(feature_range=(0,1)), MinMaxScaler(feature_range=(-1,1))
scaler0to1, scaler1to1 = scaler0to1.fit(dataset), scaler1to1.fit(dataset)
norm0to1, norm1to1 = scaler0to1.transform(dataset), scaler1to1.transform(dataset)

# Standardize (or Z-score) the data
scaler = StandardScaler()
scaler = scaler.fit(dataset)
zscore = scaler.transform(dataset)

# convert dataframe data to time series (t-1)
super0to1 = multivariate_to_supervised(norm0to1, lag=1)
super1to1 = multivariate_to_supervised(norm1to1, lag=1)
superZscore = multivariate_to_supervised(zscore, lag=1)

# superZscore = DataFrame(superZscore)
# predicting 1 timestep and 1 variable, Power_kW, drop the remaining columns after T variable of interest

super0to1 = super0to1.iloc[:,:13]
super1to1 = super1to1.iloc[:,:13]
superZscore = superZscore.iloc[:,:13]
# print(superZscore.head(4), print(superZscore.shape))

# split into train and test sets

train01, test01, train11, test11 = super0to1.values[:-10037], super0to1.values[-10037:], super1to1.values[:-10037], super1to1.values[-10037:]
trainZ, testZ = superZscore.values[:-10037], superZscore.values[-10037:]

print(train01.shape, test01.shape, trainZ.shape, testZ.shape)

# Separate into inputs and outpouts
train01X, train01y = train01[:,:-1], train01[:,-1]
test01X, test01y = test01[:,:-1], test01[:,-1]
train11X, train11y = train11[:,:-1], train11[:,-1]
test11X, test11y = test11[:,:-1], test11[:,-1]
trainZX, trainZy = trainZ[:,:-1], trainZ[:,-1]
testZX, testZy = testZ[:,:-1], testZ[:,-1]

# reshape into 3D input [samples, timesteps, features]
train01X3D = train01X.reshape((train01X.shape[0], 1, train01X.shape[1]))
test01X3D = test01X.reshape((test01X.shape[0], 1, test01X.shape[1]))
train11X3D = train11X.reshape((train11X.shape[0], 1, train11X.shape[1]))
test11X3D = test11X.reshape((test11X.shape[0], 1, test11X.shape[1]))
trainZX3D = trainZX.reshape((trainZX.shape[0], 1, trainZX.shape[1]))
testZX3D = testZX.reshape((testZX.shape[0], 1, testZX.shape[1]))

print(testZX3D.shape, train11X3D.shape)


X = ([trainZX3D],[train11X3D],[train01X3D])
y = ([trainZy],[train11y],[train01y])
X_test =([testZX3D],[test11X3D],[test01X3D])
y_test = ([testZy],[test11y],[test01y] )
actFx = ('tanh','tanh','sigmoid')

# X = ([train11X3D],[train01X3D])
# y = ([train11y],[train01y])
# X_test =([test11X3D],[test01X3D])
# y_test = ([test11y],[test01y])
names = ('Z-score','Normalized -1 to 1','Normalized 0 to 1')
epochs = 15
batch_size=24
train = DataFrame()
val = DataFrame()

for a,b,c,d,e,f  in zip(actFx,X,y, X_test, y_test, names):
    #print(a,b,c,d)
    model = Sequential()
    #model.add(LSTM(30, activation=a, batch_input_shape=(batch_size, train01X3D.shape[1], train01X3D.shape[2]), stateful=True))
    model.add(LSTM(100, activation=a, input_shape=(train01X3D.shape[1], train01X3D.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(b,c, epochs=epochs, validation_data=(d, e), batch_size=batch_size, verbose=1, shuffle=False)
    #model.reset_states()
    
    train[f] = history.history['loss']
    #val[(x for x in ds_names)] = history.history['val_loss']
    val[f] = history.history['val_loss']

# plot train and validation loss
pyplot.figure(figsize=(8,8))
for i,b in zip (range(len(names)),names):
    x = range(1,len(train) +1)
    #pyplot.figure(figsize=(12,10))
    pyplot.subplot(3,1,(i+1))
    pyplot.plot(x,train.iloc[:,i], color='blue', label='test')
    pyplot.plot(x,val.iloc[:,i], color='orange', label='validation')
    pyplot.title('model - train loss vs validation loss: %s' % (b)) 
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.xticks(x)
    pyplot.legend()
pyplot.tight_layout()
pyplot.show()

# box plot of validation sets
val.boxplot()
pyplot.title(' Validation Loss ')
pyplot.show()
print(val.describe())
