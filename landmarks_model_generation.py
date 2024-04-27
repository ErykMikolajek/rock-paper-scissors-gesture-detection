import csv

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = 'model/landmarks_dataset.csv'
model_save_path = 'model/gestures_classifier'

classes = ['rock', 'paper', 'scissors']


print()
model = keras.Sequential(
	keras.layers.InputLayer(),
	keras.layers.Dense()
)