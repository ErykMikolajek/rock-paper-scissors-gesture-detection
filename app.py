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
    left_player_score = 0
    right_player_score = 0
    player_gestures = {0: [], 1: []}
    show_gestures = 80
    announce_winner = False
    text = ''


    while True:
        fps = fps_calc.get()

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            mode = 1 - mode
        print(mode)
        # number, mode = select_mode(key, mode)

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
        cv2.line(org_image, (y // 2, 0), (y // 2, x), (0, 0, 0), 5)
        if show_gestures == 0 or mode == 1:
            if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 2:
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
                    # logging_csv(number, mode, pre_processed_landmark_list)

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

                    if hand_landmarks.landmark[0].x * y < y // 2:
                        player_gestures[0] = hand_gesture
                    else:
                        player_gestures[1] = hand_gesture

                    # if handedness.classification[0].label == 'Left':
                    #     player_gestures[0] = hand_gesture
                    # else:
                    #     player_gestures[1] = hand_gesture

                print(player_gestures[0], player_gestures[1])

                # if len(player_gestures[0]) > 0 and len(player_gestures[1]) > 0:
                if show_gestures % 100 == 0 or mode == 0:
                    winner = define_winner(player_gestures[0], player_gestures[1])
                    print(winner)
                    if winner == 1:
                        left_player_score += 1
                        text = 'Left Player Wins'
                        cv2.putText(org_image, text, (y // 2 - 150, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(org_image, text, (y // 2 - 150, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    elif winner == 2:
                        right_player_score += 1
                        text = 'Right Player Wins'
                        cv2.putText(org_image, text, (y // 2 - 150, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(org_image, text, (y // 2 - 150, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    elif winner == 0:
                        text = 'Draw'
                        cv2.putText(org_image, text, (y // 2 - 50, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(org_image, text, (y // 2 - 50, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        text = 'Undefined'
                        cv2.putText(org_image, text, (y // 2 - 50, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(org_image, text, (y // 2 - 50, x // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    announce_winner = True

        org_image = draw_info(org_image, fps, mode, 10)
        # big red counter
        if show_gestures > 0:
            cv2.putText(org_image, f'Time: {show_gestures // 20}', (y // 2 - 50, x // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(org_image, f'Time: {show_gestures // 20}', (y // 2 - 50, x // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_gestures -= 1
        cv2.putText(org_image, f'Left player score: {left_player_score}', (10, x - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(org_image, f'Left player score: {left_player_score}', (10, x - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(org_image, f'Right player score: {right_player_score}', (y - 350, x - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(org_image, f'Right player score: {right_player_score}', (y - 350, x - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition', org_image)

        if announce_winner:
            cv2.waitKey(2000)
            show_gestures = 80
            player_gestures = {0: [], 1: []}
            announce_winner = False
            text = ''


    cap.release()
    cv2.destroyAllWindows()
