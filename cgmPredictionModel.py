# quick evaluation of glocose level slope prediction
# prediction based on 10 previous data points and current time of day
# largly based on https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import sys
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
from functools import reduce

# using tf-nightly build for prototype tflite model converter
print(tf.__version__)

# build data set with 10 previous known data points


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# build and train LSTM


def get_model():
    # create dataframe from 90 days of CGM readings
    df = pd.read_csv("BG_with_time.csv", names=['Time', 'BG'])

    # convert HH:MM:SS to minute of the day (0-1440)
    df['Time'] = pd.to_timedelta(df['Time'])
    df['Time'] = df['Time'].dt.total_seconds() / 60
    df['Time'] = df['Time'].astype(int)

    # print dataframe head for varification
    print(df.head(30))

    # LSTM Model
    time_steps = 10

    # Split data for training/validation
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    print("Train and Test array sizes: ")
    print(len(train), len(test))

    X_train, y_train = create_dataset(train, train.BG, time_steps)
    X_test, y_test = create_dataset(test, test.BG, time_steps)
    #print(X_train.shape, y_train.shape)

    fullDayData = df.iloc[23448:24023]
    x_fullDay, y_fullDay = create_dataset(
        fullDayData, fullDayData.BG, time_steps)
    print("Full day cgm readings for testing: ")
    print("Full day type: " + x_fullDay.dtype)
    print("Full day data: ")
    print(x_fullDay)

    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    history = model.fit(X_train, y_train, epochs=200, batch_size=16,
                        validation_split=0.1, verbose=1, shuffle=False)

    # plotting performance
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()

    # run prediction
    y_pred = model.predict(X_test)

    # plot training data and test results
    pyplot.plot(np.arange(0, len(y_train)), y_train, 'g', label="history")
    pyplot.plot(np.arange(len(y_train), len(y_train) + len(y_test)),
                y_test, marker='.', label="true")
    pyplot.plot(np.arange(len(y_train), len(y_train) + len(y_test)),
                y_pred, 'r', label="prediction")

    # plotting of "full day" evaluation
    # pyplot.plot(np.arange(len(y_train), len(y_train) + len(y_fullDay)), y_pred, 'r', label="prediction")

    pyplot.ylabel('Value')
    pyplot.xlabel('Time Step')
    pyplot.legend()
    pyplot.show()

    return model


# build model and plot
model = get_model()

# convert model to tflite for use on microcontroller
# currently not yet functional see: https://www.tensorflow.org/lite/guide/roadmap
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

open("model.tflite", "wb").write(tflite_model)

# using tinymlgen for converting to C
# tinymlgen: https://github.com/eloquentarduino/tinymlgen
# c_code = port(model, pretty_print = True)
# print(c_code)


# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tflite_model = converter.convert()


# # NN Model (can be converted using tflite currently)
# # NN Model not accurate enough currently

# # split into train, validation, test
# TRAIN_SPLIT =  int(0.6 * SAMPLES)
# TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)
# x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
# y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# # create a NN with 2 layers of 16 neurons
# model = tf.keras.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_validate, y_validate))

#testInput_pred = model.predict(x_validate)

# pyplot.plot(testInput_pred)
# pyplot.show()

# pyplot.plot(np.arange(23448, 24023), tester, 'g', label="history")
# pyplot.plot(np.arange(23458, 24023), y_pred, 'r', label="prediction")
# pyplot.ylabel('Value')
# pyplot.xlabel('Time Step')
# pyplot.legend()
# pyplot.show();
