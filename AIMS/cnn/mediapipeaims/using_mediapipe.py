import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Model path
MODEL_PATH = "E:\AIMS\AIMS\cnn\mediapipeaims\hand_landmarker.task"

# Create options
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)

# Create landmarker
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        wrist = hand[0]
        index_tip = hand[4]

        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y

        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"

        cv2.putText(frame, direction, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)


    cv2.imshow("Hand Landmarks (Tasks API)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
