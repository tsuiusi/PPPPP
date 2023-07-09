import nltk
from nltk.stem.lancaster import LancasterStemmer
import ssl
# # Disabling the SSL for the nltk download to work
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context


# nltk.download('punkt')
from nltk.tokenize import word_tokenize

stemmer = LancasterStemmer()

# nltk is used to ohe the data
# so if it doesn't work try stemming without nltk or using some other for of sentiment analysis

import numpy as np
import tensorflow as tf
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)


try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:
    # Preprocessing

    words = []
    labels = []
    # x and y so each pattern can correspond to a specific intent. (sentiment)
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Essentially splitting the string using spaces, but with nltk builtin functions
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag']) 

    # How many words the model has seen; sorted set of words
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    # Sorted labels
    labels = sorted(labels)

    # Create bag of words - one hot encoding to see what words are used in what sentences
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # Create training data by one hot encoding everything
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        # One hot encoding of words, bag is the one hot encoded list
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# Define the input layer with the appropriate shape
input_layer = tf.keras.layers.Input(shape=(len(training[0]),))

# Define the hidden layers with the appropriate number of neurons and activation function
hidden_layer_1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(32, activation='relu')(hidden_layer_1)
hidden_layer_3 = tf.keras.layers.Dense(16, activation='relu')(hidden_layer_2)

# Define the output layer with Softmax activation function
output_layer = tf.keras.layers.Dense(len(output[0]), activation='softmax')(hidden_layer_3)

# Define the model with input and output layers
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model with appropriate loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

try:
    model.load('learning')
except:
    # Training time
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save("learning")

# Making predictions:
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for w in s_words:
        for i, j in enumerate(words):
            if w == j:
                bag[i] == 1
    
    return np.array(bag)

def chat():
    print('looks like someone stabbed makise kurisu')

    while True:
        inp = input('You: ')
        if inp.lower() == "quit":
            break

        bagged = tf.reshape(bag_of_words(inp, words),shape=(1,67))
        result = model.predict([bagged])
        indexed_results = np.argmax(result)
        print(result, indexed_results)
        tag = labels[indexed_results]
        for tg in data['intents']: 
            if tg['tag'] == tag:
                responses = tg['responses']

        print(f'bot: {random.choice(responses)}')

chat()

# Bot's a bit stupid.
# Everything registers as a goodbye