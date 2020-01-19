#Imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import \
train_test_split
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)

#Create a model
model = Sequential()
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'sgd', loss = 'mse')

#Simulate data
x = np.linspace(-0.5, 2.5, 1000).reshape(-1, 1)
y = x**3 - 3*x**2 + x + 1
x_train, x_test, y_train, y_test = \
train_test_split(x, y)

#Fit the model and evaluate
model.fit(x_train, y_train, epochs = 500)
print(model.evaluate(x_test, y_test))
print(model.get_weights())
