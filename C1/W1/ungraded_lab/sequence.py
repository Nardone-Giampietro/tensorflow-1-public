import tensorflow as tf
from keras import models, layers
import numpy as np
from matplotlib import pyplot as plt

x_train = np.array([-1., 0., 1., 2., 3., 4.], dtype=float).reshape(6, 1)
y_train = np.array([-3., -1., 1., 3., 5., 7.], dtype=float).reshape(6, 1)

model = models.Sequential()

model.add(layers.Dense(units=50, activation="relu", input_shape=(1,)))
model.add(layers.Dense(units=50, activation="relu"))
model.add(layers.Dense(units=1, activation="linear"))

model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

epochs = 500

history = model.fit(x_train, y_train, epochs=epochs, verbose=0)

x_test = np.linspace(0, 20, 1000).reshape(1000, 1)
y_test = model.predict(x_test).reshape(1000, 1)
y = 2. * x_test - 1.

plt.plot(x_test, y, "-r")
plt.plot(x_test, y_test, "ob")
plt.show()

