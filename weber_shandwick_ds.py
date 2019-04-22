# -*- coding: utf-8 -*-
#!/usr/bin/python

""" Develop sentence analysis to model specific tweets
1. Build 3 Way classifier to predict sentiments of tweets
2. Evaluate your model's performance after training
3. Use the model build in task 1 to predict  sentiment of tweets in the evaluate set
4. Compare predictions you made in task3 and the evaluation set
5 Lay out the steps and justify them as if you were presenting them to your manager.
 - In task 4, are your predictions good or bad? How can you tell? - If they're good, 
 what steps will you need to take implement your model in a production setting? - 
 If they're bad, what steps will you take to improve them? How will you know they have improved enough?"""

import numpy as np, pandas as pd, re, matplotlib.pyplot as plt, seaborn as sns, nltk, string
import warnings, itertools
warnings.filterwarnings('ignore', category=DeprecationWarning)
from nltk.tokenize import word_tokenize
from utils import *
%matplotlib inline
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping




# import data sets
train = pd.read_csv('/data/train.csv', encoding='utf-8')
evaluate = pd.read_csv('/data/evaluate.csv', encoding='utf-8')

# inspect the data, look for any patterns, anomalies 
print(train.head(10), evaluate.head(10))

# identify data types, if null values exist
train.info()
evaluate.info()

print(train.describe(), evaluate.describe()) #.isnull() # insights into dependent_var

# shape of the data
print(train.shape, evaluate.shape)

# identify distributions of variables
print(train.sentiment.value_counts())
print(evaluate.sentiment.value_counts())


# append df, preprocess together, postprocess back to  individual train/evaluate sets
combine = train.append(evaluate, ignore_index=True).apply(lambda x: x.astype(str).str.lower())  # make text lower_case

# verify success
print(combine.head(), combine.tail(), combine.shape)


def clean_content(tweets):
    ''' script to parse out punctuation, numbers, special chars, short words
    , repeated letters, white spaces, @handles and urls'''
    
    # create new field to keep clean content, remove @user reference
    combine['clean_content'] = combine['content'].apply(lambda x: re.sub(r'@\w+', ' ',x))
    # remove rt| cc
    combine['clean_content'] = combine['clean_content'].apply(lambda x: re.sub('rt|cc', ' ',x))
    # remove url related info, replace with URL
    combine['clean_content'] = combine['clean_content'].apply(lambda x: re.sub(r'((www\.[\S]+)|(https?://[\S]+))', 'URL',x))
    # remove puntucation 
    combine['clean_content'] = combine['clean_content'].apply(lambda x: re.sub('[^a-zA-Z]', ' ',x))
    combine['clean_content'] = combine['clean_content'].apply(lambda x: re.sub('[#]', ' ',x))
    # remove repeating letters, i.e funnnny => funny
    combine['clean_content'] = combine['clean_content'].apply(lambda x: re.sub(r'(.)\1+', r'\1\1', x))
    # remove whitespaces
    combine['clean_content'] = combine['clean_content'].apply(lambda x: re.sub('  ', ' ', x))
    # remove short words 
    combine['clean_content'] = combine['clean_content'].apply(lambda x: ' '.join([t for t in x.split() if len(t)>3]))
    
    return tweets


clean_content(combine)[:3]

def remove_stopwords(tweets):
	''' short script to remove stopwords using the stemming function'''
    tokenized_twt = combine['clean_content'].apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokenized_twt = tokenized_twt.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming function
    # put tokens back together
    for i in range(len(tokenized_twt)):
        tokenized_twt[i] = ' '.join(tokenized_twt[i])

    combine['clean_content'] = tokenized_twt
    return(tweets)


remove_stopwords(combine)[:7]


# numeric encode the dependent variable; sentiment
sentiment_dic = {'positive':1, "neutral": 0, 'negative':-1 }

# create Series column for sentiment integer
combine['sentiment_label'] = combine['sentiment'].map(sentiment_dic)
combine.head(3)


# separate data back into train and evaluate sets
train = combine.iloc[:160000,-2:]
evaluate = combine.iloc[160000:,-2:]
print(train.shape, evaluate.shape)


def write_to_file(df):
	'''script write cleaned_text to csv file for later use'''
    df.to_csv('df.csv',sep='\t', encoding='utf-8')
    
    return df


tt = write_to_file(train)
hh = write_to_file(evaluate)


def separate_train_data(df_train):

    train_data = train.values
	y = train['sentiment_label'].values
	sentences = train['clean_content'].values
	y_rnn = keras.utils.to_categorical(y, num_classes=3)
	

	# identify max # of words in a tweet
	combine['token_length'] = [len(x.split(' ')) for x in combine.clean_content]
	maxlen = max(combine.token_length)

	# split of strings of text into individual tokens
	tokenizer = Tokenizer(num_words=12000, oov_token = True)
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)

	# set vectors to process in the network
	X_pad = pad_sequences(sequences, maxlen=maxlen)

	# split into train and validation set
	X_train_seq, X_val_seq, y_trainRNN, y_valRNN = train_test_split(X_pad, y_rnn, random_state=7, test_size=0.2)

	# split into train and validation set logistic regression
	X_train, X_val, y_train, y_val = train_test_split(sequences, y, random_state=7, test_size=0.2)

	
	return X_train_seq, X_val_seq, y_trainRNN, y_valRNN, X_train, X_val, y_train, y_val


X_train_seq, X_val_seq, y_trainRNN, y_valRNN, X_train, X_val, y_train, y_val = separate_train_data(train)


# confirm n-dims
print(X_train_seq.shape, X_val_seq.shape, y_train.shape, y_val.shape, y_trainRNN.shape, y_valRNN.shape)


seed = 7
np.random.seed(seed)


# Establish baseline models, fit the models on train set, evaluate on validation set

# Build logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train_seq, y_train)
lr_score = classifier.score(X_val_seq, y_val)
lr_yhat_val = classifier.predict(X_val_seq)
print('Accuracy for tweet sentiment with Logistic Regression: {:.3f}%'.format(lr_score*100))

# plot confusion matrix
plt.figure(1, figsize=(5,5))
cm = confusion_matrix(y_val, lr_yhat_val)
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Logisitic Regression Confusion Matrix - Validation Set')
plt.xlabel('true label')
plt.ylabel('predicted label')


# Build Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_seq, y_train)
bayes_score = model.score(X_val_seq, y_val)
bayes_yhat_val = model.predict(X_val_seq)
print('Accuracy for tweet sentiment with Naive Bayes: {:.3f}%'.format(bayes_score*100))

# plot confusion matrix and classification report
plt.figure(2, figsize=(5,5))
cm = confusion_matrix(y_val, bayes_yhat_val)
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Navies Bayes Confusion Matrix - Validation')
plt.xlabel('true label')
plt.ylabel('predicted label')
print(classification_report(y_val,bayes_yhat_val)) 
print('Naive Bayes - Validation Set:',accuracy_score(y_val, bayes_yhat_val))

# Random Forest 
RF_classifier = RandomForestClassifier(n_estimators=250, random_state=7)
RF_clf = RF_classifier.fit(X_train_seq, y_train)
RF_norm_pred = RF_clf.predict(X_val_seq)

# Plot confusion and classification report
RF_cm = confusion_matrix(y_val,RF_norm_pred)
plt.figure(3, figsize=(5,5))
sns.heatmap(RF_cm.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Random Forest Confusion Matrix - Validation Set')
plt.xlabel('true label')
plt.ylabel('predicted label')
print(classification_report(y_val,RF_norm_pred))  
print('Random Forest accuracy - Validation Set:',accuracy_score(y_val, RF_norm_pred))

# Gradient Boosting
grad_classifier = GradientBoostingClassifier(n_estimators=250, random_state=7)
grad_clf = grad_classifier.fit(X_train_seq, y_train)
grad_pred = grad_clf.predict(X_val_seq)

Grad_cm = confusion_matrix(y_val,grad_pred)
plt.figure(4, figsize=(5,5))
sns.heatmap(Grad_cm.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Adaboost Gradient Confusion Matrix - Validation Set')
plt.xlabel('true label')
plt.ylabel('predicted label')
print(classification_report(y_val,grad_pred))  
print('Gradient boostint accuracy: ', accuracy_score(y_val, grad_pred)) 


# define baseline FF model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_dim=40, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return base_model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

estimator = KerasClassifier(build_fn=base_model, epochs=6, batch_size=128, verbose=0)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X_train_seq, y_train , cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# create a plot of the history
def plot_hist(history):
    ''' visually monitor train vs validation accuracy and loss '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(x, acc, 'b', label='Training accuracy')
    plt.plot(x, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Train and Validation Loss')
    plt.legend()


def lstm_net(vocab_size, embedding_size, maxlen):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=maxlen))
    model.add(Bidirectional(LSTM(512, dropout=0.55, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    optimizer = Adam(lr=0.0003, beta_1=0.91)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    es = [EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=4, verbose=1)]

    return model_lstm


print(model.summary())

# train the model

batch_size = 128
num_epochs = 15
callbacks=[EarlyStopping(patience=4, monitor='val_loss'),
           ModelCheckpoint(filepath=save_dir + "/" + 'my_model_twitter_analysis.{epoch:02d}-{val_loss:.2f}.hdf5',\
                           monitor='val_loss', verbose=0, mode='auto', period=2)]

save_dir = 'save/NLP'
def fit_lstm(model_lstm, X_train_seq, y_trainRNN, batch_size, num_epochs, save_dir, callbacks):
    history = model.fit(X_smote, y_trainRNN, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val_seq, y_valRNN),
                    callbacks=callbacks, verbose=1, shuffle=False)

    # save the model
    save(save_dir + "/" + 'my_model_generate_sentences.h5')
    
    return history


plot_hist(history)


def run_evaluation_set(evaluate):
	'''separate evaluation data set into independent and dependent var, tokenize and pad sequences
	for models.  returns yhat prediction'''

	# process evaluation data for network
	evaluate_data = evaluate.values
	X_eval = evaluate_data[:,0]
	y_eval = evaluate_data[:,1]
	y_evalRNN = keras.utils.to_categorical(y_eval, num_classes=3)
	

	# apply the same tokenizer from train set on the evaluate set 
	sequences = tokenizer.texts_to_sequences(X_eval)
	# set vectors to process in the network
	X_eval_seq = pad_sequences(sequences, maxlen=40)

    # predict on evaluation data
	y_hat = model.predict(X_eval_seq, verbose=1)

	return y_eval, y_evalRNN, X_eval_seq, y_hat


y_eval, y_evalRNN, X_eval_seq, y_hat =run_evaluation_set(evaluate)

# inverse sentiment matrix back to scalar
y_orig = [np.argmax(y, axis=None, out=None) for y in y_evalRNN]

def lstm_accuracy(y_orig,y_hat):
    count =0
    for i in range(len(y_orig)):
        if y_orig[i] == y_hat[i]:
            count += 1
        count
    return count/y_hat.shape[0]

Print('LSTM accuracy: ',lstm_accuracy(y_orig,y_hat))


# add confusion, classification report

# process evaluation data for remaining models

lr_yhat_eval = classifier.predict(X_eval_seq)
bayes_yhat_eval = model.predict(X_eval_seq)
RF_eval_pred = RF_clf.predict(X_eval_seq)
grad_eval_pred = grad_clf.predict(X_eval_seq)

print('Classifiction Report for Logistic Regerssion - Test set: ',classification_report(y_eval, lr_yhat_eval))
print('Logistic Regeression - Test Set:',accuracy_score(y_eval, r_yhat_eval))

print('Classifiction Report for Gradient Boosting - Test set: ',classification_report(y_eval, bayes_yhat_eval))
print('Naive Bayes accuracy - Test Set:',accuracy_score(y_eval, bayes_yhat_eval))


print('Classifiction Report for Random Forest - Test set: ',classification_report(y_eval, RF_eval_pred))
print('Random Forest accuracy - Test Set:',accuracy_score(y_eval, RF_eval_pred))

print('Classifiction Report for Gradient Boosting - Test set: ',classification_report(y_eval, RF_eval_pred))
print('Gradient Boosting accuracy - Test Set:',accuracy_score(y_eval, RF_eval_pred))






