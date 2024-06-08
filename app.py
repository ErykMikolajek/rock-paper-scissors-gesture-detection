import cv2
import mediapipe as mp
import copy
import numpy as np
from utils import *

from model.gesture_classifier_model import GestureClassifierModel


def select_mode(key, mode):
    ret_number = -1
    ret_mode = mode
    if 48 <= key <= 59:  # 0 ~ 4
        ret_number = key - 48
    if key == 110:  # n
        ret_mode = 0
    if key == 107:  # k
        ret_mode = 1
    return ret_number, ret_mode


if __name__ == '__main__':

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    model_path = 'model/gesture_classifier-v01'
    gesture_classifier_model = GestureClassifierModel(model_path)

    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )

    fps_calc = FpsCalc(buffer_len=10)
    mode = 0
    use_bound_rect = True

    while True:
        fps = fps_calc.get()

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)
        org_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        x, y, _ = image.shape
        # cv2.line(org_image, (y // 2, 0), (y // 2, x), (0, 0, 0), 5)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                bound_rect = calc_bounding_rect(org_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(org_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_gesture, gesture_probability = gesture_classifier_model(pre_processed_landmark_list)

                # Drawing part
                org_image = draw_bounding_rect(use_bound_rect, org_image, bound_rect)
                org_image = draw_info_text(
                    org_image,
                    bound_rect,
                    handedness,
                    gesture_classifier_model.classes[hand_gesture],
                    gesture_probability
                )

                mp_drawing.draw_landmarks(org_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        org_image = draw_info(org_image, fps, mode, number)

        cv2.imshow('Hand Gesture Recognition', org_image)

    cap.release()
    cv2.destroyAllWindows()
