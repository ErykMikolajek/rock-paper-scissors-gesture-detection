from collections import deque
import numpy as np
import copy
import cv2
import itertools
import csv
from tensorflow.keras import backend as K


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/landmarks_dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


def draw_bounding_rect(use_bound_rect, image, bound_rect):
    if use_bound_rect:
        # Outer rectangle
        cv2.rectangle(image, (bound_rect[0], bound_rect[1]), (bound_rect[2], bound_rect[3]), (0, 0, 0), 3)
    return image


def draw_info_text(image, bound_rect, handedness, hand_gesture_text, hand_gesture_probability):
    cv2.rectangle(image, (bound_rect[0], bound_rect[1]), (bound_rect[2], bound_rect[1] - 22),
                  (0, 0, 0), -1)
    hand_gesture_text = hand_gesture_text.upper()
    handedness = handedness.classification[0].label[0:]
    hand_gesture_probability = round(hand_gesture_probability, 2)
    if hand_gesture_text != "":# and hand_gesture_probability > 0.55:
        # info_text = hand_gesture_text + ", " + str(hand_gesture_probability) + ", " + handedness
        info_text = hand_gesture_text + ", " + handedness
    else:
        info_text = "UNDEFINED" + ", " + handedness
    cv2.putText(image, info_text, (bound_rect[0] + 5, bound_rect[1] - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_DUPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_DUPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # TODO: change line below
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
    return image

def define_winner(left_player_gesture, right_player_gesture):
    #['rock', 'paper', 'scissors', 'lizard', 'spock']
    if left_player_gesture == right_player_gesture: # Draw
        return 0
    elif left_player_gesture == 0 and right_player_gesture == 1: # Rock vs Paper
        return 2
    elif left_player_gesture == 0 and right_player_gesture == 2: # Rock vs Scissors
        return 1
    elif left_player_gesture == 0 and right_player_gesture == 3: # Rock vs Lizard
        return 1
    elif left_player_gesture == 0 and right_player_gesture == 4: # Rock vs Spock
        return 2
    
    elif left_player_gesture == 1 and right_player_gesture == 0: # Paper vs Rock
        return 1
    elif left_player_gesture == 1 and right_player_gesture == 2: # Paper vs Scissors
        return 2
    elif left_player_gesture == 1 and right_player_gesture == 3: # Paper vs Lizard
        return 2
    elif left_player_gesture == 1 and right_player_gesture == 4: # Paper vs Spock
        return 1
    
    elif left_player_gesture == 2 and right_player_gesture == 0: # Scissors vs Rock
        return 2
    elif left_player_gesture == 2 and right_player_gesture == 1: # Scissors vs Paper
        return 1
    elif left_player_gesture == 2 and right_player_gesture == 3: # Scissors vs Lizard
        return 1
    elif left_player_gesture == 2 and right_player_gesture == 4: # Scissors vs Spock
        return 2
    
    elif left_player_gesture == 3 and right_player_gesture == 0: # Lizard vs Rock
        return 2
    elif left_player_gesture == 3 and right_player_gesture == 1: # Lizard vs Paper
        return 1
    elif left_player_gesture == 3 and right_player_gesture == 2: # Lizard vs Scissors
        return 2
    elif left_player_gesture == 3 and right_player_gesture == 4: # Lizard vs Spock
        return 1
    
    elif left_player_gesture == 4 and right_player_gesture == 0: # Spock vs Rock
        return 1
    elif left_player_gesture == 4 and right_player_gesture == 1: # Spock vs Paper
        return 2
    elif left_player_gesture == 4 and right_player_gesture == 2: # Spock vs Scissors
        return 1
    elif left_player_gesture == 4 and right_player_gesture == 3: # Spock vs Lizard
        return 2
    
    else: # Undefined
        return -1


class FpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._diff_times = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._diff_times.append(different_time)

        fps = 1000.0 / (sum(self._diff_times) / len(self._diff_times))
        fps_rounded = round(fps, 2)

        return fps_rounded
