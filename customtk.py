#importing required modules
import os
import subprocess
import threading
import time
import tkinter
import customtkinter
from PIL import ImageTk,Image
import cv2
import mediapipe as mp
import numpy as np
import copy
import itertools
from collections import deque, Counter
import multiprocessing as mup
import tensorflow as tf
import pydirectinput

# import threading

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  #creating cutstom tkinter window
app.geometry("12500x800")
app.maxsize(1250,800)
app.title('G-tech Login')
app.iconbitmap('icon.ico')
message = ["lewlhada"]
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils  
start_time = 0

actions = ['stop', 'goLeft', 'goRight', 'modeDiaPo', 'modeNormal']
poses = ['left-right', 'up-down', 'stop']


def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)                     # Mirror display
    debug_image = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results ,debug_image

def pre_process_landmark(landmark_list):
    temporarly_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temporarly_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temporarly_landmark_list[index][0] = temporarly_landmark_list[index][0] - base_x
        temporarly_landmark_list[index][1] = temporarly_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temporarly_landmark_list = list(
        itertools.chain.from_iterable(temporarly_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temporarly_landmark_list)))

    def normalize_(n):
        return n / max_value

    temporarly_landmark_list = list(map(normalize_, temporarly_landmark_list))

    return temporarly_landmark_list


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_Keypoint_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image
def draw_info(image, mode, number):
    if 1 <= mode <= 2:
        cv2.putText(image, "Collecting Data ", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                   cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33, 225, 225), 1,
                       cv2.LINE_AA)
    return image
def draw_info_text(image, brect,finger_gesture_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    
    # info_text = handedness.classification[0].label[0:]
        
    if finger_gesture_text != "":
        cv2.putText(image, "Hand Gesture:" + finger_gesture_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "Hand Gesture:" + finger_gesture_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv2.LINE_AA)
        # info_text = info_text + ':' + finger_gesture_text
    # cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0]!= 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image
def draw_landmarks(image, landmark_point):
    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # 人差指
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # 中指
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # 薬指
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # 小指
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # 手の平
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1) 
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h] # retuen the coords of 4 points of the rect

class PoseClassifier(object):
    def __init__(
        self,
        model_path='Sign_classifier_MetaData.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index


Pose_classifier = PoseClassifier()
class Classifier(object):
    def __init__(
        self,
        model_path='Gestures_classifier_MetaData.tflite',
        score_th=0.8,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index



keypoint_classifier = Classifier()

##################################################################   
history_length = 16  # lenght of list that takes max indexes of predections
Keypoints_history = deque(maxlen=history_length)
Argmax_list = deque(maxlen=history_length)
use_boundary_recttangle = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 390)
# most_common_fg_id = None
ActionDetected = 0





def Ai():
    # code for the Ai() function goes here
    # with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:

    #     while cap.isOpened():
    #         key = cv2.waitKey(10)
    #         if key == 27:
    #             break
    #         if stopevent.is_set():
    #             break

    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # Make detections
    #         image, results, debug_image = mediapipe_detection(frame, hands)
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
    #                                                     results.multi_handedness):

    #                 # Landmark calculation
    #                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)

    #                 # Conversion to relative coordinates / normalized coordinates
    #                 pre_processed_landmark_list = pre_process_landmark(
    #                     landmark_list)
    #                 pre_processed_Keypoints_list = pre_process_Keypoint_history(
    #                     debug_image, Keypoints_history)

    #                 hand_id = Pose_classifier(pre_processed_landmark_list)

    #                 hand_sign_id = 0
    #                 hand_sign_len = len(pre_processed_Keypoints_list)

    #                 if hand_sign_len == (history_length * 2):
    #                         hand_sign_id = keypoint_classifier(
    #                             pre_processed_Keypoints_list)

    #                 Argmax_list.append(hand_sign_id)
    #                 most_common_fg_id = Counter(
    #                     Argmax_list).most_common()
    #                 if hand_id in (0, 1):
    #                     landmark_index = 8 if hand_id == 0 else 12
    #                     Keypoints_history.append(landmark_list[landmark_index])
    #                     action_detected = [1, 2] if hand_id == 0 else [3, 4]
    #                     ActionDetected = most_common_fg_id[0][0] if most_common_fg_id[0][0] in action_detected else 0
    #                     # message = actions[ActionDetected]
    #                     first_time = True
    #                     # time.sleep(5)
                        
    #                     if (message[-1] != actions[ActionDetected]):
    #                         print(message[-1])
    #                         message.append(actions[ActionDetected])
    #                         if message[-1] == "goLeft":
    #                             pydirectinput.press("left")
    #                         elif message[-1] == "goRight":
    #                             pydirectinput.press("right")
    #                         elif message[-1] == "goUp":
    #                             pydirectinput.press("up")
    #                         elif message[-1] == "goDown":
    #                             pydirectinput.press("down")

    #                 else:
    #                     Keypoints_history.append([0, 0])
    #                     ActionDetected = 0

    #             cv2.imshow('Hand Gesture Recognition', debug_image)


    # cap.release()

    # cv2.destroyAllWindows()
    # pass
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:

        while cap.isOpened():

            # Process Key (ESC: end)
            key = cv2.waitKey(10)
            if key == 27:  # ESC
                break
            if stopevent.is_set():
               break
            # number, mode = select_mode(key, mode)

            # Camera capture #####################################################
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results, debug_image = mediapipe_detection(frame, hands)
            ActionDetected = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    
                    if handedness.classification[0].label == "Right" :
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)

                        # Landmark calculation
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        pre_processed_Keypoints_list = pre_process_Keypoint_history(
                            debug_image, Keypoints_history)

                        hand_id = Pose_classifier(pre_processed_landmark_list)

                        hand_sign_id = 0
                        hand_sign_len = len(pre_processed_Keypoints_list)

                        if hand_id in (0, 1):
                            landmark_index = 8 if hand_id == 0 else 12
                            Keypoints_history.append(landmark_list[landmark_index])
                        else:
                            Keypoints_history.append([0, 0])
                            
                        if hand_sign_len == (history_length * 2):
                            hand_sign_id = keypoint_classifier(
                                pre_processed_Keypoints_list)
                            
                        action_detected = [1, 2] if hand_id == 0 else [3, 4]
                        Argmax_list.append(hand_sign_id)
                        most_common_fg_id = Counter(
                            Argmax_list).most_common()
                        if most_common_fg_id[0][0] in action_detected:
                            ActionDetected = most_common_fg_id[0][0]

                        if (message[-1] != actions[ActionDetected]):
                            print(message[-1])
                            message.append(actions[ActionDetected])
                            if message[-1] == "goLeft":
                                pydirectinput.press("left")
                            elif message[-1] == "goRight":
                                pydirectinput.press("right")
                            elif message[-1] == "goUp":
                                pydirectinput.press("up")
                            elif message[-1] == "goDown":
                                pydirectinput.press("down")
                        

                            
                        # Drawing part
                        debug_image = draw_bounding_rect(
                            use_boundary_recttangle, debug_image, brect)
                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            brect,
                            actions[ActionDetected],
                        )
            else:
                Keypoints_history.append([0, 0])

            # debug_image = draw_info(debug_image, mode, number)
            debug_image = draw_point_history(debug_image, Keypoints_history)

            # Screen reflection
            # cv2.imshow('Hand Gesture Recognition', debug_image)


    cap.release()
    cv2.destroyAllWindows()
    pass


thread = threading.Thread(target=Ai)
stopevent = threading.Event()

def button_function():
    app.destroy()            # destroy current window and creating new one 
    w = customtkinter.CTk()  
    w.geometry("1280x720")
    update = False 
    w.maxsize(1250,725)
    w.minsize(1250,725)
    w.title('G-tech Home')
    w.iconbitmap('icon.ico')

    label = customtkinter.CTkLabel(master=w, text='')
    label.pack(pady=20, padx=20)
    # l1=customtkinter.CTkLabel(master=w, text="Home Page",font=('Century Gothic',60))
    # l1.place(relx=0.5, rely=0.5,  anchor=tkinter.CENTER)

    # camera.place_forget()


    frame=customtkinter.CTkFrame(master=w, width=900, height=690, corner_radius=15,fg_color='#1F1E3F')
    frame.place(relx=0.625, rely=0.5, anchor=tkinter.CENTER)

    logo = ImageTk.PhotoImage(Image.open("icon.png"))
    logo_label = customtkinter.CTkLabel(master=frame, image=logo, text="")
    logo_label.place(relx=0.009, rely=0.009, anchor=tkinter.NW)

    # camera = tkinter.Label(master=frame, width=500, height=390, text='',borderwidth=2, relief="solid")
    nocamera = customtkinter.CTkFrame(master=frame, width=500, height=390, corner_radius=15,fg_color='#2E2B55', border_color='#F5D115', border_width=2)

    nocamera.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)

    nocameralabel = customtkinter.CTkLabel(master=nocamera, text="Launch Your Experience", font=('Century Gothic', 30, 'bold'))
    nocameralabel.place(relx=0.5, rely=0.5,  anchor=tkinter.CENTER)
    camera = customtkinter.CTkFrame(master=frame, width=250, height=190, )
    
    # Center the text horizontally

    def show_frame():
        # camera.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)
        nocamera.place_forget()
        thread.start()
        
        
        update = True

    def close():
        # camera.place_forget()
        nocamera.place(relx=0.5, rely=0.4,  anchor=tkinter.CENTER)
        stopevent.set()

        # w.destroy()

    buttonconn = customtkinter.CTkButton(master=frame, text="Launch ", hover_color='#FF4500',font=('Helvetica', 16, 'bold'), border_color='#F5D115', border_width=2, command=show_frame )
    buttonconn.configure(width=300, height=50, corner_radius=6, fg_color='#29284E', text_color='#8465C7' , hover_color='#362067')
  # Prevent the label from shrinking
    buttonconn.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)

    buttondeconn = customtkinter.CTkButton(master=frame, text="Close ", hover_color='#FF4500',font=('Helvetica', 16, 'bold'), border_color='#F5D115', border_width=2, command=close )
    buttondeconn.configure(width=300, height=50, corner_radius=6, fg_color='#29284E', text_color='#8465C7' , hover_color='#362067')
    # Prevent the label from shrinking
    buttondeconn.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)


    sidebar = customtkinter.CTkFrame(master=w, width=200, height=670, fg_color='#1F1E3F',  corner_radius=15)
    sidebar.place(relx=0.01, rely=0.023, anchor=tkinter.NW)
    # sidebar.pack(side=tkinter.LEFT)

    # Add widgets to the sidebar
    label1 = customtkinter.CTkLabel(master=sidebar, text="AirWavecontrol", font=('Century Gothic', 15, 'bold'))
    label1.pack(pady=50, padx=45)

    button1 = customtkinter.CTkButton(master=sidebar, text="MS Power Point", hover_color='#FF4500',font=('Helvetica', 16, 'bold') )
    button1.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='orange' , hover_color='#FF4500', )

    button1.pack(pady=10, padx=0)

    button2 = customtkinter.CTkButton(master=sidebar, text="Netflix", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button2.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button2.pack(pady=5, padx=0)

    button3 = customtkinter.CTkButton(master=sidebar, text="spotify", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button3.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button3.pack(pady=5, padx=0)

    button4 = customtkinter.CTkButton(master=sidebar, text="keynote", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button4.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button4.pack(pady=5, padx=0)

    button5 = customtkinter.CTkButton(master=sidebar, text="Zoom", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button5.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button5.pack(pady=5, padx=0)

    button6 = customtkinter.CTkButton(master=sidebar, text="Windows Os", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button6.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button6.pack(pady=5, padx=0)

    button7 = customtkinter.CTkButton(master=sidebar, text="Web Browser", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button7.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button7.pack(pady=5, padx=0)

    button8 = customtkinter.CTkButton(master=sidebar, text="Virtual env", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button8.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button8.pack(pady=5, padx=0)

    # Load the image files for the logos
    instagram_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(os.path.dirname(__file__), './ig.png')), size=(30, 30))
    twitter_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(os.path.dirname(__file__), './twit.png')), size=(30, 30))
    gmail_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(os.path.dirname(__file__), './fb.png')), size=(30, 30))

    # Create labels for each logo
    instagram_label = customtkinter.CTkLabel(master=sidebar, image=instagram_image, bg_color='transparent', text='')
    twitter_label = customtkinter.CTkLabel(master=sidebar, image=twitter_image, bg_color='transparent', text='')
    gmail_label = customtkinter.CTkLabel(master=sidebar, image=gmail_image, bg_color='transparent', text='')

    # Pack the labels horizontally
    instagram_label.pack(side=tkinter.LEFT, padx=20, pady=20)
    twitter_label.pack(side=tkinter.LEFT, padx=20,)
    gmail_label.pack(side=tkinter.LEFT, padx=(20,0))

    name = customtkinter.CTkLabel(master = sidebar,text="G-Tech",font=('Century Gothic',15, 'bold') )
    name.pack(pady=(15,10), padx=(1,5))
    # l1=customtkinter.CTkLabel(master=w, ))

    cap = cv2.VideoCapture(0)



    # def update_video():
    #     _, frame = cap.read()
    #     frame = cv2.flip(frame, 1)
    #     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #     img = Image.fromarray(cv2image)
    #     imgtk = ImageTk.PhotoImage(image=img)
    #     camera.imgtk = imgtk
    #     camera.configure(image=imgtk)
    #     camera.after(1, update_video) # Schedule the next update

    # update_video()
    w.mainloop()
    


img1=ImageTk.PhotoImage(Image.open("pattern.jpg"))
l1=customtkinter.CTkLabel(master=app,image=img1)
l1.pack()

#creating custom frame
frame=customtkinter.CTkFrame(master=l1, width=320, height=360, corner_radius=15)
frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

l2=customtkinter.CTkLabel(master=frame, text="Log into your Account",font=('Century Gothic',20))
l2.place(x=50, y=45)

entry1=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Username')
entry1.place(x=50, y=110)

entry2=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Password', show="*")
entry2.place(x=50, y=165)

l3=customtkinter.CTkLabel(master=frame, text="Forget password?",font=('Century Gothic',12))
l3.place(x=155,y=195)

#Create custom button
button1 = customtkinter.CTkButton(master=frame, width=220, text="Login", command=button_function, corner_radius=6)
button1.place(x=50, y=240)


img2=customtkinter.CTkImage(Image.open("Google__G__Logo.svg.webp").resize((20,20), Image.ANTIALIAS))
img3=customtkinter.CTkImage(Image.open("124010.png").resize((20,20), Image.ANTIALIAS))
button2= customtkinter.CTkButton(master=frame, image=img2, text="Google", width=100, height=20, compound="left", fg_color='white', text_color='black', hover_color='#AFAFAF')
button2.place(x=50, y=290)

button3= customtkinter.CTkButton(master=frame, image=img3, text="Facebook", width=100, height=20, compound="left", fg_color='white', text_color='black', hover_color='#AFAFAF')
button3.place(x=170, y=290)




# You can easily integrate authentication system 



app.mainloop()
