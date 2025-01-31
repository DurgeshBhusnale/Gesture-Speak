import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',24: 'Y',25: 'Z',26:' '
}

# Initialize variables
gesture_start_time = None
recognized_alphabets = []
alphabet_counts = {label: 0 for label in labels_dict.values()}
prev_recognized_alphabet = None

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            data_aux = []

            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

               

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Recognize alphabet
            prediction = model.predict([np.asarray(data_aux)])
            recognized_alphabet = labels_dict[int(prediction[0])]
            alphabet_counts[recognized_alphabet] += 1

            # Draw bounding box and recognized alphabet
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, recognized_alphabet, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            # Start gesture recognition timer
            if gesture_start_time is None:
                gesture_start_time = time.time()



    # Check if 1 second has passed
    if gesture_start_time is not None and time.time() - gesture_start_time >= 2:
        # Find most frequent recognized alphabet
        most_common_alphabet = max(alphabet_counts, key=alphabet_counts.get)
        if alphabet_counts[most_common_alphabet] >= 5:
            # recognized_alphabet.append(most_common_alphabet)
            # print("Recognized word:", recognized_alphabet)
            recognized_alphabets.append(most_common_alphabet)
            word = ''.join(recognized_alphabets)
            print("Recognized word:", word)
        
        # Reset variables
        alphabet_counts = {label: 0 for label in labels_dict.values()}
        gesture_start_time = None

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




