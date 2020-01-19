#Imports
import scipy.io as sio, numpy as np, pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

#Don’t forget to set a seed!
np.random.seed(123)

#Data Preparation, from Kaggle
def get_data(): 
    mnist = sio.loadmat('mnist-original.mat') # not shuffled
    return mnist['data'].T, mnist['label'][0]

def preprocess(X, y):
    X = X.astype(float) / 255
    labels = to_categorical(y, num_classes = None)
    shuffle_index = np.random.permutation(len(X))
    X, labels = X[shuffle_index], labels[shuffle_index]
    return (X[:60000], labels[:60000],
        X[60000:], labels[60000:])

#Create the model
def create_model():
    model = Sequential()
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model

def compile_and_train(model, X, y):
    model.compile(optimizer = 'adam',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
    model.fit(X, y, epochs = 5, batch_size = 32)

def main():
    X, y = get_data()
    X_train, y_train, X_test, y_test = preprocess(X, y)
    model = create_model()
    compile_and_train(model, X_train, y_train)
    print(model.evaluate(X_test, y_test))

if __name__ == '__main__':
    main()
