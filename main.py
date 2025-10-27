import cv2
import numpy as np
import mediapipe as mp


# Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

# Define landmark indices for both eyes
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]

class FaceEffect:
    DEBUG = 0
    BIG_EYE = 1
    FACE_AUGMENTATION = 2
    MOTION_TRACKING = 3


# Functions
def get_landmarks(image):
    """Detect face landmarks with MediaPipe Face Mesh"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if result.multi_face_landmarks:
        return result.multi_face_landmarks[0]
    return None


def get_eye_polygon(landmarks, eye_points, w, h):
    """Return polygon (numpy array) for an eye given landmark indices"""
    pts = np.array(
        [[int(landmarks.landmark[i].x * w),
          int(landmarks.landmark[i].y * h)] for i in eye_points]
    )
    return pts

def apply_big_eye_effect(image, eye_pts, scale=1.2):
    """
    Extreme big eyes
    """
    x, y, w, h = cv2.boundingRect(eye_pts)
    pad_w = int(0.4 * w)
    pad_h = int(0.4 * h)
    x1 = max(x - pad_w, 0)
    y1 = max(y - pad_h, 0)
    x2 = min(x + w + pad_w, image.shape[1])
    y2 = min(y + h + pad_h, image.shape[0])
    roi = image[y1:y2, x1:x2].copy()

    h_r, w_r = roi.shape[:2]
    if h_r < 10 or w_r < 10:
        return image

    # strong radial bulge
    cx, cy = w_r / 2, h_r / 2
    y_grid, x_grid = np.indices((h_r, w_r), dtype=np.float32)
    x_norm = (x_grid - cx) / cx
    y_norm = (y_grid - cy) / cy
    r = np.sqrt(x_norm**2 + y_norm**2)
    r = np.minimum(r, 1.0)
    k = (scale - 1.0) * 3.0   # exaggerate bulge
    new_r = r / (1 + k * (1 - r**2))
    map_x = cx + x_norm / (r + 1e-6) * new_r * cx
    map_y = cy + y_norm / (r + 1e-6) * new_r * cy
    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # very wide mask
    mask = np.zeros((h_r, w_r, 3), np.float32)
    center = (int(cx), int(cy))
    axes = (int(w_r * 0.7), int(h_r * 0.7))
    cv2.ellipse(mask, center, axes, 0, 0, 360, (1, 1, 1), -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 30)

    blended = (roi * (1 - mask) + warped * mask).astype(np.uint8)
    image[y1:y2, x1:x2] = blended
    return image

def fingers_up(hand_landmarks, handedness):
    # returns [thumb, index, middle, ring, pinky] bools
    tips = [4, 8, 12, 16, 20]
    pip = [2, 6, 10, 14, 18]  # use lower joint for comparison
    coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    handed = handedness.classification[0].label if handedness else "Right"
    up = [False]*5

    # For fingers except thumb: tip.y < pip.y -> finger up (image origin top-left)
    for i in range(1,5):
        up[i] = coords[tips[i]][1] < coords[pip[i]][1]

    # Thumb: use x comparison depending on handedness
    if handed == "Right":
        up[0] = coords[tips[0]][0] > coords[pip[0]][0]
    else:
        up[0] = coords[tips[0]][0] < coords[pip[0]][0]

    return up

def detect_gesture(up):
    cnt = sum(up)
    if cnt == 0:
        return "Fist"
    if cnt == 5:
        return "Open"
    if up == [False, True, False, False, False]:
        return "Point"
    if up[0] and not any(up[1:]):
        return "Thumb"
    return f"{cnt} fingers"


# Main Loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    print("Press ESC or Q to quit.")

    faceEffect = FaceEffect.BIG_EYE

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        key = cv2.waitKey(5) & 0xFF
        # Exit on 'q' or ESC
        if key in (27, ord('q')):
            break
        # Apply big eye effect
        elif key == ord('0'):
            faceEffect = FaceEffect.DEBUG
        elif key == ord('1'):
            faceEffect = FaceEffect.BIG_EYE
        elif key == ord('2'):
            faceEffect = FaceEffect.FACE_AUGMENTATION
        elif key == ord('3'):
            faceEffect = FaceEffect.MOTION_TRACKING

        
        if faceEffect == FaceEffect.DEBUG:
            landmarks = get_landmarks(frame)
            if landmarks:
                pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark])
                x1, y1, w1, h1 = cv2.boundingRect(pts)
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

                # Draw eye bounding boxes
                left_eye_pts = get_eye_polygon(landmarks, LEFT_EYE, w, h)
                x2, y2, w2, h2 = cv2.boundingRect(left_eye_pts)
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                right_eye_pts = get_eye_polygon(landmarks, RIGHT_EYE, w, h)
                x3, y3, w3, h3 = cv2.boundingRect(right_eye_pts)
                cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), 2)

        elif faceEffect == FaceEffect.BIG_EYE:
            landmarks = get_landmarks(frame)

            if landmarks:
                # Task 1: Face Warp
                left_eye = get_eye_polygon(landmarks, LEFT_EYE, w, h)
                right_eye = get_eye_polygon(landmarks, RIGHT_EYE, w, h)

                frame = apply_big_eye_effect(frame, left_eye, scale=1.2)
                frame = apply_big_eye_effect(frame, right_eye, scale=1.2)
                
        elif faceEffect == FaceEffect.FACE_AUGMENTATION:
            pass

        elif faceEffect == FaceEffect.MOTION_TRACKING:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    up = fingers_up(hand_landmarks, handedness)
                    gesture = detect_gesture(up)

                    # position label near wrist landmark (0)
                    x = int(hand_landmarks.landmark[0].x * w)
                    y = int(hand_landmarks.landmark[0].y * h) - 20
                    cv2.putText(frame, f"{gesture} ({handedness.classification[0].label})",
                                (x, max(y, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Real-time Face Effects", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

