import cv2
import os
import time
import math
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---------- CONFIG ----------
MODEL_PATH = "E:\AIMS\AIMS\cnn\mediapipeaims\hand_landmarker.task"
SAVE_DIR = "dataset"
STABILITY_FRAMES = 8
SAVE_COOLDOWN = 0.5
MAX_IMAGES = 500
# ----------------------------

# Create folders
directions = ["up", "down", "left", "right"]
for d in directions:
    os.makedirs(os.path.join(SAVE_DIR, d), exist_ok=True)

# Count existing images
counts = {}
for d in directions:
    counts[d] = len(os.listdir(os.path.join(SAVE_DIR, d)))

print("Initial counts:", counts)

# MediaPipe setup
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

last_direction = None
stable_count = 0
last_save_time = 0

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
        thumb_tip = hand[4]

        dx = thumb_tip.x - wrist.x
        dy = thumb_tip.y - wrist.y

        angle = math.degrees(math.atan2(dy, dx))

        if -45 <= angle <= 45:
            direction = "right"
        elif 45 < angle <= 135:
            direction = "down"
        elif -135 <= angle < -45:
            direction = "up"
        else:
            direction = "left"

        # Stability check
        if direction == last_direction:
            stable_count += 1
        else:
            stable_count = 0

        last_direction = direction

        # Save condition
        if (
            stable_count >= STABILITY_FRAMES
            and (time.time() - last_save_time) > SAVE_COOLDOWN
            and counts[direction] < MAX_IMAGES
        ):
            filename = f"{counts[direction]}.jpg"
            path = os.path.join(SAVE_DIR, direction, filename)
            cv2.imwrite(path, frame)

            counts[direction] += 1
            last_save_time = time.time()
            stable_count = 0

            print(f"{direction} → {counts[direction]}/{MAX_IMAGES}")

    # Display counts on screen
    y_offset = 80
    for d in directions:
        text = f"{d.upper()}: {counts[d]}/{MAX_IMAGES}"
        cv2.putText(frame, text, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += 30

    cv2.imshow("Auto Dataset Generator", frame)

    # Stop if all classes full
    if all(counts[d] >= MAX_IMAGES for d in directions):
        print("Dataset complete.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
