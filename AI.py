# Camera preparation
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Set mediapipe model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:

    while cap.isOpened():

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
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
                            if message[-2] == 'stop':
                                if message[-1] == "goLeft":
                                    pydirectinput.press("left")
                                elif message[-1] == "goRight":
                                    pydirectinput.press("right")
                                elif message[-1] == "goUp":
                                    pydirectinput.press("up")
                                elif message[-1] == "goDown":
                                    pydirectinput.press("down")
                            else:
                                pass

                    else:
                        Keypoints_history.append([0, 0])

                    print(actions[ActionDetected])
                    # Drawing part
                    debug_image = draw_bounding_rect(
                        use_boundary_recttangle, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        actions[ActionDetected],
                    )
        else:
            Keypoints_history.append([0, 0])

        # debug_image = draw_info(debug_image, mode, number)
        debug_image = draw_point_history(debug_image, Keypoints_history)

        # Screen reflection
        cv2.imshow('Hand Gesture Recognition', debug_image)


cap.release()

cv2.destroyAllWindows()