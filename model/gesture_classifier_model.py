import numpy as np
import tensorflow as tf
from utils import f1_score


class GestureClassifierModel:
    def __init__(self, model_path):
        super().__init__()
        self.classes = ['rock', 'paper', 'scissors', 'lizard', 'spock']
        self.num_classes = len(self.classes)
        self.model = self.load_model(model_path)

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(42,)))
        self.model.add(tf.keras.layers.Dense(80, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(60, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(40, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Dense(30, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        return self.model

    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path, custom_objects={"f1_score": f1_score})
        except IOError as e:
            self.model = self.build_model()
        print("Model loaded successfully")
        return self.model

    def __call__(self, inputs):
        if self.model is None:
            raise ValueError("Model not loaded or built")
        inputs = np.array(inputs)
        if inputs.ndim < 2:
            prediction = self.model.predict(tf.expand_dims(inputs, axis=0), verbose=0)
            print(tf.nn.softmax(prediction).numpy())
            class_index = np.argmax(prediction, axis=1)[0]
            class_proba = tf.nn.softmax(prediction).numpy()[0, class_index]
            return class_index, class_proba
        else:
            prediction = self.model.predict(inputs, verbose=0)
            class_index = np.argmax(prediction, axis=1)
            return class_index

