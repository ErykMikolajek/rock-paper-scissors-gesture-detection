import tensorflow as tf

dataset = 'model/landmarks_dataset.csv'
model_save_path = 'model/gestures_classifier'

classes = ['rock', 'paper', 'scissors']

class ModelSubClassing(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.num_classes = len(classes)
        self.input = tf.keras.layers.InputLayer()
        self.dense1 = tf.keras.layers.Dense(100)
        self.dense2 = tf.keras.layers.Dense(100)
        self.dense3 = tf.keras.layers.Dense(100)
        self.output = tf.keras.layers.Dense(self.num_classes)

    def train(self, input_tensor, training=False):
        # forward pass: block 1 
        x = self.conv1(input_tensor)
        x = self.max1(x)
        x = self.bn1(x)

        # forward pass: block 2 
        x = self.conv2(x)
        x = self.bn2(x)

        # droput followed by gap and classifier
        x = self.drop(x)
        x = self.gap(x)
        return self.dense(x)
        
    def load_data(self, file_path):
        pass