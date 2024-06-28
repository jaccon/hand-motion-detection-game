import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)

star_image = cv2.imread('assets/star.png', cv2.IMREAD_UNCHANGED)
star_image = cv2.resize(star_image, (30, 30), interpolation=cv2.INTER_AREA)

def is_hand_open(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    return thumb_tip < pinky_tip

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    h, w, _ = img_overlay.shape
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)
    y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
    x1o, x2o = max(0, -x), min(w, img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    roi = img[y1:y2, x1:x2]
    img_overlay_roi = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None]

    img[y1:y2, x1:x2] = alpha * img_overlay_roi + (1 - alpha) * roi

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    hand_open_count = 0
    progress = 0
    phase = 0
    stars = 0
    hand_was_open = False

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

                if is_hand_open(hand_landmarks.landmark):
                    if not hand_was_open:
                        hand_open_count += 1
                        progress += 10  
                        hand_was_open = True

                        if hand_open_count % 10 == 0:
                            phase += 1

                            if phase % 1 == 0:
                                stars += 1
                else:
                    hand_was_open = False

        cv2.rectangle(frame, (50, frame.shape[0] - 50), (50 + progress * 5, frame.shape[0] - 30), (0, 255, 0), -1)
        cv2.rectangle(frame, (50, frame.shape[0] - 50), (550, frame.shape[0] - 30), (255, 255, 255), 2)
        cv2.putText(frame, f'Motion detected: {hand_open_count}', (50, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Level: {phase}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(stars):
            overlay_image_alpha(frame, star_image[:, :, :3], frame.shape[1] - 50, 50 + i * 40, star_image[:, :, 3] / 255.0)

        if progress >= 100:
            progress = 0
            hand_open_count = 0

        cv2.imshow('Motion detection exercise', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
