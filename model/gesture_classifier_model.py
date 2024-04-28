import numpy as np
import tensorflow as tf


class GestureClassifierModel:
    def __init__(self, model_path):
        super().__init__()
        self.classes = ['rock', 'paper', 'scissors']
        self.num_classes = len(self.classes)
        self.model = self.load_model(model_path)

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(42,)))
        self.model.add(tf.keras.layers.Dense(50, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(40, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Dense(30, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        return self.model

    def load_model(self, model_path):
        # try:
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return self.model

    def __call__(self, inputs):
        if self.model is None:
            raise ValueError("Model not loaded or built")
        inputs = np.array(inputs)
        prediction = self.model.predict(tf.expand_dims(inputs, axis=0), verbose=0)
        class_index = np.argmax(prediction, axis=1)[0]
        class_proba = tf.nn.softmax(prediction).numpy()[0, class_index]
        return class_index, class_proba
