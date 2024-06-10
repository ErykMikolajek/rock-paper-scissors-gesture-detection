# Rock-Paper-Scissors Gesture Detection

This project implements a gesture recognition program for the game Rock-Paper-Scissors, extended to include Spock and Lizard. Using a camera feed, the program recognizes gestures made by two players and tallies their scores based on the game rules.


## Installation
To get started, clone the repository and install the necessary dependencies from the requirements.txt file.
```bash
git clone https://github.com/yourusername/Rock-Paper-Scissors-Gesture-Detection.git
cd Rock-Paper-Scissors-Gesture-Detection
pip install -r requirements.txt
```
If you need to train the gesture classifier model, run the following script:
```bash
python gesture_classifier_trainer.py
```
Once the model is trained, you can start the application by running:
```bash
python app.py
```

## Usage
After running app.py, a GUI window will pop up. Position your camera to capture the hand gestures of both players. The window is split in half - each half is reserved for one player gesture. The program will recognize the gestures on the both halfs of the screen and keep score based on the rules of Rock-Paper-Scissors-Spock-Lizard.

## Model Details
The gesture recognition system uses Google MediaPipe to detect and generate landmarks for the hands. These landmarks are then normalized and passed to a small MLP (Multi-Layer Perceptron) neural network for gesture classification. The architecture of the model is 5 layers around 60 neurons each and dropout layers between each hidden layer.
The model is built using Keras' Sequential API.
