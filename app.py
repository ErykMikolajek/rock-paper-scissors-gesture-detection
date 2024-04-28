import cv2
import mediapipe as mp
import copy
import numpy as np
from utils import *


def select_mode(key, mode):
    ret_number = -1
    ret_mode = mode
    if 48 <= key <= 50:  # 0 ~ 2
        ret_number = key - 48
    if key == 110:  # n
        ret_mode = 0
    if key == 107:  # k
        ret_mode = 1
    return ret_number, ret_mode


if __name__ == '__main__':

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

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
                # hand_sign_id = landmark_classifier(pre_processed_landmark_list)

                # Finger gesture classification
                finger_gesture_id = 0

                # Drawing part
                org_image = draw_bounding_rect(use_bound_rect, org_image, bound_rect)
                # org_image = draw_landmarks(org_image, landmark_list)
                # org_image = draw_info_text(
                #     org_image,
                #     bound_rect,
                #     handedness,
                #     keypoint_classifier_labels[hand_sign_id],
                #     point_history_classifier_labels[most_common_fg_id[0][0]],
                # )

        org_image = draw_info(org_image, fps, mode, number)

        # Screen reflection #############################################################
        cv2.imshow('Hand Gesture Recognition', org_image)

    cap.release()
    cv2.destroyAllWindows()









# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     temp_point_history = copy.deepcopy(point_history)
#
#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, point in enumerate(temp_point_history):
#         if index == 0:
#             base_x, base_y = point[0], point[1]
#
#         temp_point_history[index][0] = (temp_point_history[index][0] -
#                                         base_x) / image_width
#         temp_point_history[index][1] = (temp_point_history[index][1] -
#                                         base_y) / image_height
#
#     # Convert to a one-dimensional list
#     temp_point_history = list(
#         itertools.chain.from_iterable(temp_point_history))
#
#     return temp_point_history



    # with mp_hands.Hands(
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5,
    # ) as hands:
    #     while cap.isOpened():
    #         success, img = cap.read()
    #         image = img.copy()
    #         if not success:
    #             print("Ignoring empty camera frame.")
    #             continue
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         image = cv2.flip(image, 1)
    #         image.flags.writeable = False
    #         results = hands.process(image)
    #         image.flags.writeable = True
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         x, y, _ = image.shape
    #         cv2.line(image, (y//2, 0), (y//2, x), (0, 0, 0), 5)
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
    #                 print(hand_landmarks)
    #                 if handedness.classification[0].label == 'Left':
    #                     cv2.rectangle(image, (0, 0), (50, 50), (255, 0, 0),  thickness=-1)
    #                 elif handedness.classification[0].label == 'Right':
    #                     cv2.rectangle(image, (y - 50, 0), (y, 50), (0, 0, 255), thickness=-1)
    #                 mp_drawing.draw_landmarks(
    #                     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #         cv2.imshow('MediaPipe Hands', image)
    #         if cv2.waitKey(5) & 0xFF == ord("q"):
    #             break
