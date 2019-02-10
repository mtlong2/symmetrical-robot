# sentiment analysis with LSTM
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pandas import read_csv, concat, DataFrame
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, TimeDistributed, Dropout
import matplotlib.pyplot as plt
from keras.datasets import imdb



np.random.seed(6)

# plot accuracy and loss from training and testing set
def plot_hist(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(x, acc, 'b', label='Training accuracy')
    plt.plot(x, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x, loss, 'b', label='training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Train and Validation Loss')
    plt.legend()

# import highest 5000 words
freq_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=freq_words)

# truncate and pad the sequences to only 500 words
max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# build the model
embedding_vect = 32
model = Sequential()
model.add(Embedding(freq_words, embedding_vect, input_length=max_len))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# fit the model 
history = model.fit(X_train, y_train, epochs=8, verbose=1, batch_size=96 , validation_data=(X_test,y_test))

# evaluate the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
print('training accuracy %.4f' % accuracy)
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print('testing accuracy %.4f' % accuracy)
plot_hist(history)

# make a prediction 
wrd_2_id = imdb.get_word_index()

# predict sentiment from imdb reviews
awful = 'this movie was awful give me my time back'
great = 'this movie was stellar from the start through the end'

for review in [awful,great]:
    temp = []
    for word in review.split(' '):
        temp.append(wrd_2_id[word])
    temp_padded = pad_sequences([temp],maxlen=max_len)
    print('%s. Sentiment: %s' % (review, model.predict(np.array([temp_padded][0]))))

