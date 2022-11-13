import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frames = cap.read()
        image = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        print(result.multi_hand_landmarks)





        cv2.imshow('hand',frames)
        key = cv2.waitKey(30) &0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyWindow()
