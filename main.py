import tensorflow as ts
import keras
from keras import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np


# build the network architecture
def build_network(input_size, output_size):
    net = Sequential()
    net.add(Input(shape=(input_size,)))
    net.add(Dense(8, activation='relu'))
    net.add(Dense(8, activation='relu'))
    net.add(Dense(output_size))
    # add BatchNormalization, Dropout, kernel initializer? bias initializer?
    net.summary()
    return net


if __name__ == "__main__":
    network = build_network(5,7)
    input = np.array([1,2,3,4,5])
    print(input)
