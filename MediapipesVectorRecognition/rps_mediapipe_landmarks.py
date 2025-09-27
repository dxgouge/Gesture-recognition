import mediapipe as mp
import cv2
import numpy as np
import time
import os
from pynput import keyboard

# Import MediaPipe Tasks API components
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables to store the latest result
latest_result = None
latest_annotated_image = None
running = True
listener = None


def on_press(key):
    """Handle keyboard input for quitting the application."""
    global running
    if key.char == 'q':
        running = False
        listener.stop()


def start_global_listener():
    """Start the global keyboard listener."""
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


def process_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function that processes hand landmark detection results.
    This runs in a separate thread for real-time processing.
    """
    global latest_result, latest_annotated_image
    
    try:
        # Store the latest result
        latest_result = result
        
        # Convert MediaPipe image to numpy array for OpenCV
        rgb_image = output_image.numpy_view()
        annotated_image = rgb_image.copy()
        
        # Process hand landmarks if detected
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                try:

                    x_coords = [landmark.x for landmark in hand_landmarks]
                    y_coords = [landmark.y for landmark in hand_landmarks]
                    
                    def get_landmark_distance(landmark1, landmark2):
                        """Calculate distance between two landmarks."""
                        return np.sqrt(
                            np.square(x_coords[landmark1] - x_coords[landmark2]) + 
                            np.square(y_coords[landmark1] - y_coords[landmark2])
                        )

                    def get_finger_vector(landmark1, landmark2):
                        """Calculate vector between two landmarks."""
                        initial_coords = [x_coords[landmark1], y_coords[landmark1]]
                        terminal_coords = [x_coords[landmark2], y_coords[landmark2]]
                        return np.array([
                            terminal_coords[0] - initial_coords[0], 
                            terminal_coords[1] - initial_coords[1]
                        ])

                    def get_finger_angle(vector):
                        """Calculate angle of a vector in degrees."""
                        return np.degrees(np.arctan2(vector[1], vector[0]))

                    def get_angle_between_fingers(vector1, vector2):
                        """Calculate angle between two vectors in degrees."""
                        dot_product = np.dot(vector1, vector2)
                        norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                        return np.degrees(np.arccos(dot_product / norms))

                    # Calculate distances from finger tips to base
                    dis_index_tip_to_base = get_landmark_distance(8, 0)
                    dis_middle_tip_to_base = get_landmark_distance(12, 0)
                    dis_ring_tip_to_base = get_landmark_distance(16, 0)
                    dis_pinky_tip_to_base = get_landmark_distance(20, 0)
                    
                    # Get reference coordinates
                    static_coords = [x_coords[9], y_coords[9]]
                    base_coords1 = [x_coords[0], y_coords[0]]

                    # Calculate direction vectors based on landmarks. From lower joint to middle joint.
                    dirV_index = get_finger_vector(7, 6)
                    dirV_middle = get_finger_vector(11, 10)
                    dirV_ring = get_finger_vector(15, 14)
                    dirV_pinky = get_finger_vector(19, 18)
                    dirV_baseline1 = get_finger_vector(1, 0)
                    
                    dirV_middle_base = get_finger_vector(10, 9)
                    dirV_ring_base = get_finger_vector(14, 13)
         

                    # Calculate angles
                    angle_index = get_finger_angle(dirV_index)
                    angle_middle = get_finger_angle(dirV_middle)
                    angle_ring = get_finger_angle(dirV_ring)
                    angle_pinky = get_finger_angle(dirV_pinky)
                    angle_baseline1 = get_finger_angle(dirV_baseline1)
                    
                    angle_middle_to_ring = get_angle_between_fingers(dirV_middle, dirV_ring)
                    angle_base_middle_to_ring = get_angle_between_fingers(dirV_middle_base, dirV_ring_base)
                    angle_middle_to_baseline1 = get_angle_between_fingers(dirV_middle, dirV_baseline1)

                    # Scale the vectors to hand size for vector smoothing
                    palm_size = get_landmark_distance(9, 0)  # 0.14
                    scale = 0.14 / palm_size
                    
                    # Apply scaling
                    dis_index_tip_to_base *= scale
                    dis_middle_tip_to_base *= scale
                    dis_ring_tip_to_base *= scale
                    dis_pinky_tip_to_base *= scale
                    angle_middle_to_ring *= scale
                    dirV_index *= scale
                    dirV_middle *= scale
                    dirV_ring *= scale
                    dirV_pinky *= scale
                    dirV_baseline1 *= scale
                    
                    
                    # Vector-based gesture recognition
                    
                    #checks if hand is pointing up or down then checks if the ring and middle finger are pointing in opposite directions
                    if (base_coords1[1] > static_coords[1] and dirV_middle_base[1] < 0 and dirV_ring_base[1] < 0):
                        gesture_type = "rock"
                    elif (base_coords1[1] < static_coords[1] and dirV_middle_base[1] > 0 and dirV_ring_base[1] > 0):
                        gesture_type = "rock"
                    elif (base_coords1[1] > static_coords[1] and dirV_middle[1] < 0 and dirV_ring[1] < 0):
                        gesture_type = "rock"
                    elif (base_coords1[1] < static_coords[1] and dirV_middle[1] > 0 and dirV_ring[1] > 0):
                        gesture_type = "rock"
                    #checks if the middle and ring finger are pointing in opposite directions
                    elif angle_middle_to_ring > 90:
                        gesture_type = "scissors"
                    #checks if the middle and ring finger are pointing in the same direction
                    elif (angle_middle - angle_ring < 90 and angle_middle - angle_ring > -90):
                        gesture_type = "paper"
                    else:
                        gesture_type = "unknown"
                    
                    # Distance-based gesture recognition (commented out)
                    # closed_finger_distance = 0.15
                    # if (dis_index_to_base < closed_finger_distance and 
                    #     dis_middle_to_base < closed_finger_distance and 
                    #     dis_ring_to_base < closed_finger_distance and 
                    #     dis_pinky_to_base < closed_finger_distance):
                    #     gesture_type = "rock"
                    # elif (dis_index_to_base > closed_finger_distance and 
                    #       dis_middle_to_base > closed_finger_distance and 
                    #       dis_ring_to_base < closed_finger_distance and 
                    #       dis_pinky_to_base < closed_finger_distance):
                    #     gesture_type = "scissors"
                    # elif (dis_index_to_base > closed_finger_distance and 
                    #       dis_middle_to_base > closed_finger_distance and 
                    #       dis_ring_to_base > closed_finger_distance and 
                    #       dis_pinky_to_base > closed_finger_distance):
                    #     gesture_type = "paper"
                    # else:
                    #     gesture_type = "unknown"

                    # Display gesture type on image
                    if gesture_type != "unknown":
                        text_x = int(min(x_coords) * annotated_image.shape[1])
                        text_y = int(min(y_coords) * annotated_image.shape[0]) - 10
                        cv2.putText(
                            annotated_image, 
                            f"{gesture_type}", 
                            (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (255, 255, 255), 
                            2
                        )
                    
                    # Draw landmarks
                    for landmark in hand_landmarks:
                        index = hand_landmarks.index(landmark)

                        landmark_x = int(landmark.x * annotated_image.shape[1])
                        landmark_y = int(landmark.y * annotated_image.shape[0])
                           
                        cv2.circle(annotated_image, (landmark_x, landmark_y), 3, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Error drawing hand bounding box: {e}")
                    continue
        
        latest_annotated_image = annotated_image
        
    except Exception as e:
        print(f"Error in process_result: {e}")
        # Fallback: just use the original image
        try:
            latest_annotated_image = output_image.numpy_view()
        except:
            latest_annotated_image = None

# Download the hand landmarker model (this will be done automatically)
# The model will be downloaded to a cache directory on first run

# Create hand landmarker options with CPU processing
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')  # This will auto-download
        # No delegate specified = uses CPU by default, GPU delegation not compatible with windows
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result,
    num_hands=2,  # Detect up to 2 hands
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("MediaPipe Tasks Hand Landmarks Started!")
print("Press 'q' to quit")

# Create the hand landmarker
with HandLandmarker.create_from_options(options) as landmarker:
    frame_count = 0
    print("CPU processing enabled!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Get current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Process the frame (this will trigger the callback function)
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # Display the latest annotated result
        if latest_annotated_image is not None:
            # Convert RGB back to BGR for OpenCV display
            display_image = cv2.cvtColor(latest_annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Tasks Hand Landmarks (CPU)', display_image)
        else:
            cv2.imshow('MediaPipe Tasks Hand Landmarks (CPU)', frame)
        
        # Print detection info every 30 frames
        frame_count += 1
        if frame_count % 30 == 0 and latest_result is not None:
            if latest_result.hand_landmarks:
                print(f"Detected {len(latest_result.hand_landmarks)} hand(s)", end="\r", flush=True)
            else:
                print("No hands detected")
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

cap.release()
cv2.destroyAllWindows()
print("MediaPipe Tasks Hand Landmarks stopped!")

