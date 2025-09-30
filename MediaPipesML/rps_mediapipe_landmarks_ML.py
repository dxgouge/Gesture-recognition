import mediapipe as mp
import cv2
import numpy as np
import time
import os
from pynput import keyboard
from scipy.spatial.distance import pdist
from typing import Optional, List, Dict, Any, Tuple

# Import MediaPipe Tasks API components
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

start_time = time.time()
class HandGestureRecognizer:
    """
    A class for real-time hand gesture recognition using MediaPipe.
    Supports Rock, Paper, Scissors gesture detection.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 camera_index: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 data_collection_mode: bool = False):
        """
        Initialize the HandGestureRecognizer.
        
        Args:
            model_path: Path to the MediaPipe hand landmarker model
            camera_index: Camera index for video capture
            frame_width: Frame width for video capture
            frame_height: Frame height for video capture
        """
        self.data_collection_mode = data_collection_mode
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # State variables
        self.latest_result: Optional[HandLandmarkerResult] = None
        self.latest_annotated_image: Optional[np.ndarray] = None
        self.running = True
        self.listener: Optional[keyboard.Listener] = None
        self.landmarker: Optional[HandLandmarker] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Data storage for the latest processed frame
        self.latest_data: Optional[Dict[str, Any]] = None
        self.hand_detected = False
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest processed hand gesture data.
        
        Returns:
            Dictionary containing:
            - timestamp: Current timestamp in milliseconds
            - gesture_type: Recognized gesture ("rock", "paper", "scissors", "unknown")
            - x_coords: List of x coordinates for all 21 landmarks
            - y_coords: List of y coordinates for all 21 landmarks
            - angles: Dictionary of calculated angles
            - distances: Dictionary of calculated distances
            - vectors: Dictionary of calculated direction vectors
            - palm_size: Size of the palm
            - scale_factor: Scaling factor applied
        """
        return self.latest_data
    
    def _on_press(self, key):
        """Handle keyboard input for quitting the application."""
        try:
            if hasattr(key, 'char') and key.char == 'q':
                self.running = False
                if self.listener:
                    self.listener.stop()
        except AttributeError:
            pass
    
    def _start_keyboard_listener(self):
        """Start the global keyboard listener."""
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
    
    def _get_landmark_distance(self, landmark1: int, landmark2: int, x_coords: List[float], y_coords: List[float]) -> float:
        """Calculate distance between two landmarks."""
        return np.sqrt(
            np.square(x_coords[landmark1] - x_coords[landmark2]) + 
            np.square(y_coords[landmark1] - y_coords[landmark2])
        )
    
    def _get_finger_vector(self, landmark1: int, landmark2: int, x_coords: List[float], y_coords: List[float]) -> np.ndarray:
        """Calculate vector between two landmarks."""
        initial_coords = [x_coords[landmark1], y_coords[landmark1]]
        terminal_coords = [x_coords[landmark2], y_coords[landmark2]]
        return np.array([
            terminal_coords[0] - initial_coords[0], 
            terminal_coords[1] - initial_coords[1]
        ])
    
    def _get_finger_angle(self, vector: np.ndarray) -> float:
        """Calculate angle of a vector in degrees."""
        return np.degrees(np.arctan2(vector[1], vector[0]))
    
    def _get_angle_between_fingers(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        dot_product = np.dot(vector1, vector2)
        norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norms == 0:
            return 0
        return np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))
    
    def _recognize_gesture(self, x_coords: List[float], y_coords: List[float], 
                          dirV_middle: np.ndarray, dirV_ring: np.ndarray,
                          dirV_middle_base: np.ndarray, dirV_ring_base: np.ndarray,
                          angle_middle: float, angle_ring: float, angle_middle_to_ring: float) -> str:
        """Recognize gesture based on hand landmarks and vectors."""
        static_coords = [x_coords[9], y_coords[9]]
        base_coords1 = [x_coords[0], y_coords[0]]
        
        # Vector-based gesture recognition
        # Check if hand is pointing up or down then check if the ring and middle finger are pointing in opposite directions
        if (base_coords1[1] > static_coords[1] and dirV_middle_base[1] < 0 and dirV_ring_base[1] < 0):
            return "rock"
        elif (base_coords1[1] < static_coords[1] and dirV_middle_base[1] > 0 and dirV_ring_base[1] > 0):
            return "rock"
        elif (base_coords1[1] > static_coords[1] and dirV_middle[1] < 0 and dirV_ring[1] < 0):
            return "rock"
        elif (base_coords1[1] < static_coords[1] and dirV_middle[1] > 0 and dirV_ring[1] > 0):
            return "rock"
        # Check if the middle and ring finger are pointing in opposite directions
        elif angle_middle_to_ring > 90:
            return "scissors"
        # Check if the middle and ring finger are pointing in the same direction
        elif (angle_middle - angle_ring < 90 and angle_middle - angle_ring > -90):
            return "paper"
        else:
            return "unknown"
    
    def _store_latest_data(self, timestamp: int, gesture_type: str, palm_size: float, scale: float,
                          x_coords: List[float], y_coords: List[float], angles: Dict[str, float],
                          distances: Dict[str, float],distancesAggregated: float, vectors: Dict[str, np.ndarray]):
        """Store the latest processed hand gesture data."""
        self.latest_data = {
            'timestamp': timestamp,
            'gesture_type': gesture_type,
            'x_coords': x_coords.copy(),
            'y_coords': y_coords.copy(),
            'angles': angles.copy(),
            'distances': distances.copy(),
            'distancesAggregated': distancesAggregated,
            'vectors': {key: vector.copy() for key, vector in vectors.items()},
            'palm_size': palm_size,
            'scale_factor': scale
        }
    def is_hand_detected(self):
        return self.hand_detected
    def _process_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
        Callback function that processes hand landmark detection results.
        This runs in a separate thread for real-time processing.
        """
        
        try:
            # Store the latest result
            self.latest_result = result
            
            # Convert MediaPipe image to numpy array for OpenCV
            rgb_image = output_image.numpy_view()
            annotated_image = rgb_image.copy()
            
            # Process hand landmarks if detected
            if result.hand_landmarks:
                self.hand_detected = True
                for hand_landmarks in result.hand_landmarks:
                    try:
                        x_coords = [landmark.x for landmark in hand_landmarks]
                        y_coords = [landmark.y for landmark in hand_landmarks]
                        
                        # Calculate distances from finger tips to base
                        distances = {
                            'dis_index_tip_to_base': self._get_landmark_distance(8, 0, x_coords, y_coords),
                            'dis_middle_tip_to_base': self._get_landmark_distance(12, 0, x_coords, y_coords),
                            'dis_ring_tip_to_base': self._get_landmark_distance(16, 0, x_coords, y_coords),
                            'dis_pinky_tip_to_base': self._get_landmark_distance(20, 0, x_coords, y_coords)
                        }
                        
                        # Calculate direction vectors based on landmarks. From lower joint to middle joint.
                        vectors = {
                            'dirV_index': self._get_finger_vector(7, 6, x_coords, y_coords),
                            'dirV_middle': self._get_finger_vector(11, 10, x_coords, y_coords),
                            'dirV_ring': self._get_finger_vector(15, 14, x_coords, y_coords),
                            'dirV_pinky': self._get_finger_vector(19, 18, x_coords, y_coords),
                            'dirV_baseline1': self._get_finger_vector(1, 0, x_coords, y_coords),
                            'dirV_middle_base': self._get_finger_vector(10, 9, x_coords, y_coords),
                            'dirV_ring_base': self._get_finger_vector(14, 13, x_coords, y_coords)
                        }
                        
                        # Calculate angles
                        angles = {
                            'angle_index': self._get_finger_angle(vectors['dirV_index']),
                            'angle_middle': self._get_finger_angle(vectors['dirV_middle']),
                            'angle_ring': self._get_finger_angle(vectors['dirV_ring']),
                            'angle_pinky': self._get_finger_angle(vectors['dirV_pinky']),
                            'angle_baseline1': self._get_finger_angle(vectors['dirV_baseline1']),
                            'angle_middle_to_ring': self._get_angle_between_fingers(vectors['dirV_middle'], vectors['dirV_ring']),
                            'angle_base_middle_to_ring': self._get_angle_between_fingers(vectors['dirV_middle_base'], vectors['dirV_ring_base']),
                            'angle_middle_to_baseline1': self._get_angle_between_fingers(vectors['dirV_middle'], vectors['dirV_baseline1'])
                        }
                        
                        # Scale the vectors to hand size for vector smoothing
                        palm_size = self._get_landmark_distance(9, 0, x_coords, y_coords)
                        scale = 0.14 / palm_size if palm_size > 0 else 1.0
                        landmark_coords = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks])

                       
                        distancesAggregated = pdist(landmark_coords, metric='euclidean').mean()*scale
                        

                        # Apply scaling
                        for key in distances:
                            distances[key] *= scale
                        for key in vectors:
                            vectors[key] *= scale
                        angles['angle_middle_to_ring'] *= scale
                        
                        # Recognize gesture
                        gesture_type = self._recognize_gesture(
                            x_coords, y_coords, vectors['dirV_middle'], vectors['dirV_ring'],
                            vectors['dirV_middle_base'], vectors['dirV_ring_base'],
                            angles['angle_middle'], angles['angle_ring'], angles['angle_middle_to_ring']
                        )
                        
                        # Store latest data
                        csv_timestamp_ms = int((time.time()-start_time) )
                        self._store_latest_data(csv_timestamp_ms, gesture_type, palm_size, scale,
                                              x_coords, y_coords, angles, distances, distancesAggregated, vectors)
                        
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
                            landmark_x = int(landmark.x * annotated_image.shape[1])
                            landmark_y = int(landmark.y * annotated_image.shape[0])
                            cv2.circle(annotated_image, (landmark_x, landmark_y), 3, (255, 255, 255), 2)
                            
                    except Exception as e:
                        print(f"Error processing hand landmarks: {e}")
                        continue
            else:
                self.hand_detected = False
            self.latest_annotated_image = annotated_image
            
        except Exception as e:
            print(f"Error in process_result: {e}")
            # Fallback: just use the original image
            try:
                self.latest_annotated_image = output_image.numpy_view()
            except:
                self.latest_annotated_image = None
    
    def initialize(self):
        """Initialize the hand landmarker and camera."""
        # Create hand landmarker options with CPU processing
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._process_result,
            num_hands=1,  # Detect up to 2 hands
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Create the hand landmarker
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # Start keyboard listener
        self._start_keyboard_listener()
        
        print("MediaPipe Tasks Hand Landmarks Started!")
        print("Press 'q' to quit")
        print("Use get_latest_data() method to retrieve processed data")
    
    def run(self):
        """Run the main gesture recognition loop."""
        if not self.landmarker:
            raise RuntimeError("HandGestureRecognizer not initialized. Call initialize() first.")
        
        frame_count = 0
        print("CPU processing enabled!")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
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
                self.landmarker.detect_async(mp_image, timestamp_ms)
                
                # Display the latest annotated result

                if self.latest_annotated_image is not None and not self.data_collection_mode:
                    # Convert RGB back to BGR for OpenCV display
                    display_image = cv2.cvtColor(self.latest_annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('MediaPipe Tasks Hand Landmarks (CPU)', display_image)
                else:
                    cv2.imshow('MediaPipe Tasks Hand Landmarks (CPU)', frame)
                
                # Print detection info every 30 frames
                frame_count += 1
                if frame_count % 30 == 0 and self.latest_result is not None:
                    if self.latest_result.hand_landmarks:
                        print(f"Detected {len(self.latest_result.hand_landmarks)} hand(s)", end="\r", flush=True)
                    else:
                        print("No hands detected")
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.landmarker:
            self.landmarker.close()
        
        if self.listener:
            self.listener.stop()
        
        cv2.destroyAllWindows()
        print("MediaPipe Tasks Hand Landmarks stopped!")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    # Create and run the gesture recognizer
    with HandGestureRecognizer() as recognizer:
        recognizer.run()

