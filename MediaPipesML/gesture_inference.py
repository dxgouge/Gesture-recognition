"""
Rock Paper Scissors - Real-Time Inference
==========================================
Integrates the trained LightGBM model into the live MediaPipe pipeline.
Maintains a rolling 15-frame buffer and predicts on every N frames.

Usage:
    python rps_inference.py

Requirements:
    - rps_lgbm_model.pkl         (trained model from train_classifier.py)
    - hand_landmarker.task       (MediaPipe model)
    - rps_mediapipe_landmarks_ML.py (existing recognizer)
"""

import cv2
import time
import collections
import numpy as np
import joblib
import mediapipe as mp
from rps_mediapipe_landmarks_ML import HandGestureRecognizer

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH          = "rps_lgbm_model.pkl"   # Path to trained LightGBM model
WINDOW_SIZE         = 5                      # Must match training window size
PREDICT_EVERY_N_FRAMES = 1                    # Change to 5 to predict every 5 frames

# Columns used during training — must match DROP_COLUMNS in train_classifier.py
# These are excluded from the feature vector
DROP_COLUMNS = {"timestamp", "gesture_type"}

# Display colors per gesture (BGR)
GESTURE_COLORS = {
    "Rock":     (80,  80,  220),   # red-ish
    "Paper":    (80,  200, 80),    # green
    "Scissors": (220, 160, 50),    # blue-ish
    "Unknown":  (120, 120, 120),   # grey
}

GESTURE_NAMES = {0: "Rock", 1: "Paper", 2: "Scissors"}

# =============================================================================
# FEATURE EXTRACTION
# Mirrors the exact order used in log_data_to_csv() in the data collection file
# =============================================================================

def extract_feature_vector(data: dict) -> np.ndarray:
    """
    Convert a raw data dict from get_latest_data() into a flat feature vector.
    Order must exactly match the CSV column order used during training,
    minus the DROP_COLUMNS (timestamp, gesture_type).

    Columns (in order):
        palm_size, scale_factor,
        landmark_0_x ... landmark_20_x,
        landmark_0_y ... landmark_20_y,
        angle_index, angle_middle, angle_ring, angle_pinky,
        angle_baseline1, angle_middle_to_ring, angle_base_middle_to_ring, angle_middle_to_baseline1,
        dis_index_tip_to_base, dis_middle_tip_to_base, dis_ring_tip_to_base, dis_pinky_tip_to_base,
        distancesAggregated,
        dirV_index_x, dirV_index_y, dirV_middle_x, dirV_middle_y,
        dirV_ring_x, dirV_ring_y, dirV_pinky_x, dirV_pinky_y,
        dirV_baseline1_x, dirV_baseline1_y, dirV_middle_base_x, dirV_middle_base_y,
        dirV_ring_base_x, dirV_ring_base_y
    """
    row = [
        
        # Landmark x coords (21)
        #*data['x_coords'],
        # Landmark y coords (21)
        #*data['y_coords'],
        # Angles (8)
        data['angles']['angle_index'],
        data['angles']['angle_middle'],
        data['angles']['angle_ring'],
        data['angles']['angle_pinky'],
        data['angles']['angle_baseline1'],
        data['angles']['angle_middle_to_ring'],
        data['angles']['angle_base_middle_to_ring'],
        data['angles']['angle_middle_to_baseline1'],
        # Distances (4)
        data['distances']['dis_index_tip_to_base'],
        data['distances']['dis_middle_tip_to_base'],
        data['distances']['dis_ring_tip_to_base'],
        data['distances']['dis_pinky_tip_to_base'],
        # Aggregated distance (1)
        data['distancesAggregated'],
        # Direction vectors (7 × 2 = 14)
        data['vectors']['dirV_index'][0],    data['vectors']['dirV_index'][1],
        data['vectors']['dirV_middle'][0],   data['vectors']['dirV_middle'][1],
        data['vectors']['dirV_ring'][0],     data['vectors']['dirV_ring'][1],
        data['vectors']['dirV_pinky'][0],    data['vectors']['dirV_pinky'][1],
        data['vectors']['dirV_baseline1'][0],data['vectors']['dirV_baseline1'][1],
        data['vectors']['dirV_middle_base'][0], data['vectors']['dirV_middle_base'][1],
        data['vectors']['dirV_ring_base'][0],   data['vectors']['dirV_ring_base'][1],
    ]
    return np.array(row, dtype=np.float64)


# =============================================================================
# OVERLAY DRAWING
# =============================================================================

def draw_overlay(image: np.ndarray, gesture: str, confidence: float, buffer_size: int) -> np.ndarray:
    """Draw gesture prediction overlay onto the frame."""
    color = GESTURE_COLORS.get(gesture, GESTURE_COLORS["Unknown"])
    h, w = image.shape[:2]

    # Background pill
    cv2.rectangle(image, (10, 10), (320, 90), (20, 20, 20), -1)
    cv2.rectangle(image, (10, 10), (320, 90), color, 2)

    # Gesture name
    cv2.putText(image, gesture, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Confidence
    cv2.putText(image, f"{confidence*100:.1f}%", (200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Buffer fill indicator
    bar_w = int((buffer_size / WINDOW_SIZE) * 290)
    cv2.rectangle(image, (10, 75), (300, 85), (40, 40, 40), -1)
    cv2.rectangle(image, (10, 75), (10 + bar_w, 85), color, -1)
    cv2.putText(image, f"Buffer {buffer_size}/{WINDOW_SIZE}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    return image


# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================

def run_inference():
    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    # Rolling buffer — automatically discards frames older than WINDOW_SIZE
    frame_buffer = collections.deque(maxlen=WINDOW_SIZE)

    # Inference state
    current_gesture    = "Unknown"
    current_confidence = 0.0
    frame_count        = 0

    recognizer = HandGestureRecognizer(data_collection_mode=False)

    try:
        recognizer.initialize()
        print(f"Running inference every {PREDICT_EVERY_N_FRAMES} frame(s). Press 'q' to quit.")

        while recognizer.running:
            ret, frame = recognizer.cap.read()
            if not ret:
                print("Failed to read from webcam")
                break

            # Send frame to MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            recognizer.landmarker.detect_async(mp_image, timestamp_ms)

            frame_count += 1

            # Only process when hand is detected
            if recognizer.is_hand_detected():
                data = recognizer.get_latest_data()
                if data:
                    # Extract features and add to rolling buffer
                    features = extract_feature_vector(data)
                    frame_buffer.append(features)

                    # Predict every N frames once buffer is full
                    if (len(frame_buffer) == WINDOW_SIZE and
                            frame_count % PREDICT_EVERY_N_FRAMES == 0):

                        # Flatten window into single feature vector
                        window = np.array(frame_buffer).flatten().reshape(1, -1)

                        # Get class probabilities
                        probs = model.predict_proba(window)[0]
                        pred  = int(np.argmax(probs))

                        current_gesture    = GESTURE_NAMES[pred]
                        current_confidence = probs[pred]
            else:
                # Clear buffer when hand leaves frame
                #frame_buffer.clear()
                #current_gesture    = "Unknown"
                current_confidence = 0.0

            # Build display frame
            if recognizer.latest_annotated_image is not None:
                display = cv2.cvtColor(recognizer.latest_annotated_image, cv2.COLOR_RGB2BGR)
            else:
                display = frame

            # Draw overlay
            display = draw_overlay(display, current_gesture, current_confidence, len(frame_buffer))
            cv2.imshow("RPS Inference", display)

            # Quit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            time.sleep(0.033)  # ~30 fps

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        recognizer.cleanup()


if __name__ == "__main__":
    run_inference()