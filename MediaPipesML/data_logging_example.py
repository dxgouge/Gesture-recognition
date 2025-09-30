#!/usr/bin/env python3
"""
Example showing how to use the HandGestureRecognizer class to collect data
and save it to a CSV file in a separate logging module.
"""

import csv
import re
import cv2
import mediapipe as mp
import os
import time
from rps_mediapipe_landmarks_ML import HandGestureRecognizer
import pandas as pd

headers =         ['timestamp', 'gesture_type', 'palm_size', 'scale_factor',
        # Landmark coordinates (21 landmarks × 2 coordinates = 42 columns)
        *[f'landmark_{i}_x' for i in range(21)],
        *[f'landmark_{i}_y' for i in range(21)],
        # Vector angles (8 angles)
        'angle_index', 'angle_middle', 'angle_ring', 'angle_pinky', 
        'angle_baseline1', 'angle_middle_to_ring', 'angle_base_middle_to_ring', 'angle_middle_to_baseline1',
        # Distances (4 finger tip to base distances)
        'dis_index_tip_to_base', 'dis_middle_tip_to_base', 'dis_ring_tip_to_base', 'dis_pinky_tip_to_base',
        #Distances Aggregated
        'distancesAggregated',
        # Direction vectors (7 vectors × 2 components = 14 columns)
        'dirV_index_x', 'dirV_index_y', 'dirV_middle_x', 'dirV_middle_y', 
        'dirV_ring_x', 'dirV_ring_y', 'dirV_pinky_x', 'dirV_pinky_y', 
        'dirV_baseline1_x', 'dirV_baseline1_y', 'dirV_middle_base_x', 'dirV_middle_base_y', 'dirV_ring_base_x', 'dirV_ring_base_y'
]
    

def setup_csv_file(filename: str) -> list:
    """Setup CSV file with headers for data logging."""
    
    if not os.path.exists(filename):
     df = pd.DataFrame(columns=headers)
     df.to_csv(filename, mode='w', index= False)
    

    
    return headers





def log_data_to_csv(filename: str, data: dict, gesture_type: str):
    """Log hand gesture data to CSV file."""
    if not data:
        return
    
    # Prepare data row
    row_data = [
        data['timestamp'], gesture_type, data['palm_size'], data['scale_factor'],
        # Landmark coordinates
        *data['x_coords'], *data['y_coords'],
        # Angles
        data['angles']['angle_index'], data['angles']['angle_middle'], data['angles']['angle_ring'], 
        data['angles']['angle_pinky'], data['angles']['angle_baseline1'], data['angles']['angle_middle_to_ring'],
        data['angles']['angle_base_middle_to_ring'], data['angles']['angle_middle_to_baseline1'],
        # Distances
        data['distances']['dis_index_tip_to_base'], data['distances']['dis_middle_tip_to_base'],
        data['distances']['dis_ring_tip_to_base'], data['distances']['dis_pinky_tip_to_base'],
        # Distances Aggregated
        data['distancesAggregated'],
        # Vectors
        data['vectors']['dirV_index'][0], data['vectors']['dirV_index'][1],
        data['vectors']['dirV_middle'][0], data['vectors']['dirV_middle'][1],
        data['vectors']['dirV_ring'][0], data['vectors']['dirV_ring'][1],
        data['vectors']['dirV_pinky'][0], data['vectors']['dirV_pinky'][1],
        data['vectors']['dirV_baseline1'][0], data['vectors']['dirV_baseline1'][1],
        data['vectors']['dirV_middle_base'][0], data['vectors']['dirV_middle_base'][1],
        data['vectors']['dirV_ring_base'][0], data['vectors']['dirV_ring_base'][1]
    ]
    df = pd.DataFrame([row_data], columns=headers)
    df.to_csv(filename,mode='a', index=False, header=False)


def collect_gesture_data(duration_seconds: int = 10, output_file: str = "collected_gesture_data.csv"):
    """
    Collect gesture data for a specified duration and save to CSV.
    
    Args:
        duration_seconds: How long to collect data (in seconds)
        output_file: Output CSV filename
    """
    print(f"Setting up data collection for {duration_seconds} seconds...")
    print(f"Data will be saved to: {output_file}")
    
    # Setup CSV file
    setup_csv_file(output_file)
    
    # Create recognizer
    recognizer = HandGestureRecognizer(data_collection_mode=True)
    
    try:
        recognizer.initialize()
        
        start_time = time.time()
        frame_count = 0
        data_points_collected = 0
        
        print("Starting data collection...")
        print("Make different hand gestures (rock, paper, scissors) in front of the camera")
        print("Press Ctrl+C to stop early")

        while time.time() - start_time < duration_seconds:
            
            ret, frame = recognizer.cap.read()
            if not ret:
                print("Failed to read from webcam")
                break
        
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Get current timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Process the frame
            recognizer.landmarker.detect_async(mp_image, timestamp_ms)
            
            # Display the annotated frame (with landmarks and gesture recognition)
            if recognizer.latest_annotated_image is not None:
                # Convert RGB back to BGR for OpenCV display
                display_image = cv2.cvtColor(recognizer.latest_annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('MediaPipe Tasks Hand Landmarks (CPU)', display_image)
            else:
                # Fallback to raw frame if no annotated image available
                cv2.imshow('MediaPipe Tasks Hand Landmarks (CPU)', frame)
            
            # Get latest data and log it
            data = recognizer.get_latest_data()
            if data and recognizer.is_hand_detected():
                log_data_to_csv(output_file, data, gesture_type=gesture_type)
                data_points_collected += 1
                
                # Print progress every 30 frames
                if frame_count % 600 == 0:
                    elapsed = time.time() - start_time
                    remaining = duration_seconds - elapsed
                    print(f"Progress: {elapsed:.1f}s / {duration_seconds}s, "
                          f"Data points: {data_points_collected}, "
                          f"Current gesture: {data['gesture_type']}")
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    except Exception as e:
        print(f"Error during data collection: {e}")
    finally:
        
        cv2.destroyAllWindows()
        
        recognizer.cleanup()
        print(f"\nData collection complete!")
        print(f"Total data points collected: {data_points_collected}")
        print(f"Data saved to: {output_file}")


if __name__ == "__main__":

    outputRoot = "custom_gesture_data"
    gesture_type = input("Enter the gesture type, r,p,s (0,1,2): ")
    gesture_type = int(re.search(r'\d+', gesture_type).group())
    if isinstance(gesture_type, int):
        match (gesture_type):
            case 0:
                outputGesture = "rock"
                print("Rock")
            case 1:
                outputGesture = "paper"
                print("Paper")
            case 2:
                outputGesture = "scissors"
                print("Scissors")
            case _:
                print("Invalid gesture type")
                exit()
    else:
        print("Invalid gesture type")
        exit()
    
    outputFile = f"{outputRoot}_{outputGesture}.csv"
    
    # Collect data for 30 seconds
    collect_gesture_data(duration_seconds=10, output_file=outputFile)