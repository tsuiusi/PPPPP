import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from math import exp

data = pd.read_csv('/Users/rtty/Documents/vscode/PPPPP/transformers/digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    # Highest number in vector will be the prediction 
    # A B C
    # 0.1 0.8 0.1
    # Prediction will be B
    # np just does this operation on the list - exponentiates every element in the vector and divides all of them by the sum
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(Z):
    return (1 / (1 + np.exp(-Z))) * (1 - (1 / (1 + np.exp(-Z))))

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

def tanh_deriv(Z):
    return 2 / (np.exp(Z) + np.exp(-Z))

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def getPredictions(A3):
    return np.argmax(A3, 0)

def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def makePredictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, A3 = forwardProp(W1, b1, W2, b2, W3, b3, X)
    predictions = getPredictions(A3)
    return predictions

def testPredictions(index, W1, b1, W2, b2, W3, b3):
    currentImage = X_train[:, index, None]
    prediction = makePredictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print('Prediction: ', prediction)
    print('Label:', label)

    currentImage = currentImage.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(currentImage, interpolation='nearest')
    plt.show()

def mean_squared_error(actual, predicted):
	sum_square_error = 0.0
	for i in range(len(actual)):
		sum_square_error += (actual[i] - predicted[i])**2.0
	mean_square_error = 1.0 / len(actual) * sum_square_error
	return mean_square_error

def gradientDescent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    acc = 0.

    for i in range(iterations):
        # Forward and backprop
        Z1, A1, Z2, A2, A3 = forwardProp(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backProp(Z1, Z2, A1, A2, A3, W2, W3, X, Y)
        
        # Gets accuracy of current weights and biases
        # newacc = getAccuracy(getPredictions(A3), Y).item()

        # Updates it only if the accuracy is greater
        # Didn't work so leaving it for now
        # if newacc > acc:
        #     acc = newacc
        W1, b1, W2, b2, W3, b3 = updateParams(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)


        if i % 10 == 0:
            print('Iteration:', i)
            predictions = getPredictions(A3)
            print('Accuracy:', getAccuracy(predictions, Y))
            print('Loss:', mean_squared_error(Y, predictions))
    
    return W1, b1, W2, b2, W3, b3

def forwardProp(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = tanh(Z1)
    Z2 = W2.dot(A1) +  b2
    A2 = tanh(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, A3

def backProp(Z1, Z2, A1, A2, A3, W2, W3, X, Y):
    ohy = one_hot(Y)
    dZ3 = A3  - ohy
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * tanh_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * tanh_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3

def updateParams(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


iterations = 500

W1, b1, W2, b2, W3, b3 = gradientDescent(X_train, Y_train, 0.50, iterations)
for _ in range(10):
    r = random.randint(0, iterations)
    testPredictions(r, W1, b1, W2, b2, W3, b3)