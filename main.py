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

# Define landmark indices for both eyes
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]


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



# Main Loop

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    print("Press ctrl+C to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        landmarks = get_landmarks(frame)

        if landmarks:
            # Task 1: Face Warp (Big Eyes)
            left_eye = get_eye_polygon(landmarks, LEFT_EYE, w, h)
            right_eye = get_eye_polygon(landmarks, RIGHT_EYE, w, h)

            frame = apply_big_eye_effect(frame, left_eye, scale=1.2)
            frame = apply_big_eye_effect(frame, right_eye, scale=1.2)

        cv2.imshow("Real-time Face Effects", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

