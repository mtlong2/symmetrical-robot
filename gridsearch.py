# example demonstrating GridSearchCV for hyperparameters
# GridSearchCV hyperparameters
# scikit-learn 
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
from math import sqrt

# Function to create model, required for KerasRegressor
# create lstm model required for kerasregressor
def create_model(m_cells, lr, momentum):
    # create model
    model = Sequential()
    model.add(LSTM(m_cells, activation='tanh', input_shape=(1,12)))
    model.add(Dense(1))
    optimizer = SGD(lr=lr, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer)
    # fit model
    #model.fit(train_X3D, train_y, epochs=20, batch_size=batch_size, verbose=0)
    return model

def multivariate_to_supervised(data, lag=1, dropnan=True):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    if dropnan:
        df.dropna(inplace=True)
    return df

# hyperparameters to test
# m_cells_opt = [1, 2, 5, 10, 20, 50]
# batch_size_opts = [4, 16, 64, 512]
# lr_opts = [0.001, 0.002, 0.01, 0.05, 0.1, 0.2]
# momentum_opts = [0.0, 0.2,0.3, 0.5, 0.7, 0.8, 0.9]

m_cells_opt = [5, 10]
batch_size_opts = [64,128, 512]
lr_opts = [0.001, 0.002, 0.01]
momentum_opts = [0.2, 0.4, 0.9]
param_grid_tst = {'m_cells' : m_cells_opt, 'batch_size':batch_size_opts, 'lr':lr_opts, 'momentum' : momentum_opts}


# fix seed for reproducibility
seed = 6
np.random.seed(seed)

# load dataset (from local repository) and remove columns not needed
dataset = read_csv('PowerWeather.csv', header=0, infer_datetime_format=True, index_col='Date')
dataset = dataset.drop(columns=['Consumption_kWh', 'DHI', 'DNI'], axis=1)

values = dataset.values

# convert all data to float and Z-score the data
values = values.astype('float64')
scaler = StandardScaler()
scaler = scaler.fit(values)
zscore = scaler.transform(values)

# convert to supervised problem
supervised = multivariate_to_supervised(zscore, lag=1)
supervised = supervised.iloc[:,:13]

# split into train and test sets
train, test = supervised.values[:-10037], supervised.values[-10037:]

# separate into inputs and outputs
train_X, train_y = train[:,:-1], train[:,-1]
test_X, test_y = test[:,:-1], test[:,-1]

# reshape into 3D input 
train_X3D = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X3D = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(test_X3D.shape)

Kmodel = KerasRegressor(build_fn=create_model, epochs=6, verbose=False)
grid = GridSearchCV(estimator=Kmodel, param_grid=param_grid_tst, scoring='neg_mean_squared_error', n_jobs=-1)
grid_result = grid.fit(train_X3D,train_y)

test_loss = grid.score(test_X3D, test_y)

print('Best score: %f, Best Parameter: %s' % (grid_result.best_score_, grid_result.best_params_))





