import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

DATA = pd.read_csv('/Users/rtty/Documents/vscode/PPPPP/transformers/digit-recognizer/train.csv')

data = np.array(DATA)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0] # Labels
X_dev = data_dev[1:n] / 255. # Data corresponding to value in label. 

# Data is different to if I didn't divide by 255 since
# the numbers are bigger and therefore will be more prone to 
# errors arising (e.g incomplete grad descent)

data_train = data[100:m].T
Y_train = data_train[0] # Labels
X_train = data_train[1:n] / 255. # same same
_, m_train = X_train.shape

## Learning process code (didn't work)
# y_dev = data_dev[0]
# x_dev = data_dev[1:n]
# print(X_dev[243])
# print('--------------------------------')
# print(x_dev[243])



# so now first row is labels

# parameters: generate arrays of (784, 10) (10, 1), (10, 10), (10, 1) and fill them with random values for weights and biases
def init_params():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return w1, b1, w2, b2

def forward(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1 # dot product is multiplying the two together. stop thinking of dot product as determinant dipshit
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backprop(w2, z1, a1, a2, x, y):
    m = y.size
    onehot = one_hot_encoding(y)
    dz2 = a2 - onehot
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2) # 2 simply specifies the axis of summation (2nd, inside the array)

    dz1 = w2.T.dot(dz2) * deriv_relu(z1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1)

    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha  * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2

# Activation functions
def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    # derivative of relu is 1 if > 0, 0 if less. i was gonna do an if statement but this is simpler since t/f == 1/0
    return x > 0

def softmax(x):
    return (np.exp(x)/sum(np.exp(x)))

def one_hot_encoding(y):
    # y.size is the number of examples in the dataset, so in this case 42000
    # y.max() + 1 is 10 for the 10 output classes (0 doesn't count as a number)
    onehot = np.zeros((y.size, y.max() + 1))
    # indexing using arrays - an array from 0 to 42000, or the numer of training examples, and y is the number of label
    # for each row, go to number in column, and set it to one 
    onehot[np.arange(y.size), y] = 1
    onehot = onehot.T
    return onehot

def get_acc(predictions, y):
    return np.sum(predictions == y) / y.size

def get_predictions(a2):
    return np.argmax(a2, 0)

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backprop(w2, z1, a1, a2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i % 50 == 0:
            print(f'Iteration: {i}')
            print(f'Accuracy: {get_acc(get_predictions(a2), y)}')

    return w1, b1, w2, b2

# initialize parameters - w1, b1, w2, b2 note the shape of the data.
# forward prop - weights and biases combined with input, activation function, repeat, then one hot encode the data for output
# backprop the data so it fits and etc

def predict(x, w1, b1, w2, b2):
    _, _, _, a2 = forward(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_predictions(index, w1, b1, w2, b2):
    current_img = X_train[:, index, None]
    prediction = predict(current_img, w1, b1, w2, b2)
    label = Y_train[index]
    print(f'Prediction: {prediction}')
    print(f'Label: {label}')

    current_img = current_img.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_img, interpolation='nearest')
    plt.show()

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 2000, 0.5)
for i in range(10):
    test_predictions(random.randint(1, 42000), w1, b1, w2, b2)

