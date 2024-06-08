import tensorflow as tf
import numpy as np
from gesture_classifier_model import GestureClassifierModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import f1_score


dataset = 'landmarks_dataset.csv'


class ModelTrainer:
    def __init__(self, dataset_path, model_path):
        self.model = GestureClassifierModel(model_path)
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.x_test = None
        self.y_test = None

    def train(self, epochs, batch_size):
        x_train, x_val, y_train, y_val = self.load_data(self.dataset_path)

        model = self.model.build_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[f1_score])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

        self.model.model = model

    def save_model(self):
        self.model.model.save(self.model_path)

    def plot_results(self):
        model_prediction = self.model(self.x_test)
        print(classification_report(self.y_test, model_prediction))
        
    def load_data(self, file_path):
        x_data = np.loadtxt(file_path, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
        y_data = np.loadtxt(file_path, delimiter=',', dtype='int32', usecols=0)

        x_train, x_val_test, y_train, y_val_test = train_test_split(x_data, y_data,
                                                                    train_size=0.8, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_data, y_data,
                                                                    train_size=0.5, random_state=42)
        self.x_test = x_test
        self.y_test = y_test
        return x_train, x_val, y_train, y_val


if __name__ == '__main__':
    trainer = ModelTrainer(dataset, 'gesture_classifier-v01')
    trainer.train(500, 128)
    trainer.save_model()
    trainer.plot_results()
