# Imports

import keras
import keras.backend as K
from keras.models import Model, load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import pandas as pd
from keras.utils import plot_model
import re

def plot_2d_landmarks(img, true, pred):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img.reshape((80,120,3)))
    ax.autoscale(False)
    ax.plot(pred[:,0], pred[:,1], '+w')
    plt.plot(true[:,0], true[:,1], '.r')
    ax.axis('off')
    plt.show()

def model_predict(model, test_inputs, num_samples, output_dims):
    pred = model.predict(test_inputs)
    return np.reshape(pred, (num_samples, 28, output_dims))

def p_norm_loss(y_true, y_pred):
    return K.mean(K.pow(y_pred - y_true, 4), axis=-1)

def p_norm_loss(y_true, y_pred):
    return K.mean(K.pow(y_pred - y_true, 4), axis=-1)

def landmark_accuracy(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred) < 3.)

def landmark_accuracy_n(y_true, y_pred, n):
    diff = np.abs(y_true - y_pred)
    points = np.sum(diff, axis=2)
    mask = points < n
    return np.mean(points < n)

def landmark_loss(y_true, y_pred):
    return K.mean( K.square(y_true - y_pred) * K.sigmoid( K.abs(y_true - y_pred) - 1 ), axis=-1)
