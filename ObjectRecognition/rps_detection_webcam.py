from ultralytics import YOLO
import cv2
import numpy as np
import os


def main():
    """Main function to run the Rock-Paper-Scissors YOLO detector."""
    # Load YOLO model
    # YOLO detection model
    model_path = os.path.join(os.path.dirname(__file__), "rps_yolo_training", "rock_paper_scissors_detection", "weights", "best.pt")
    model = YOLO(model_path)
    # YOLO classification model
    # model = YOLO(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Classification", "rock_paper_scissors_robust", "weights", "best.pt"))
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Rock-Paper-Scissors YOLO Detector Started!")
    print("Press 'q' to quit")
    
    frame_count = 0
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run prediction on the frame
        results = model.predict(frame, verbose=False, conf=0.1)
        
        # Draw predictions on frame
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                num_boxes = len(boxes)
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]

                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Class: {class_name}, Confidence: {confidence:.2f}")
                    
                    # Skip scissors if multiple detections
                    if num_boxes > 1 and class_name == "Scissor":
                        continue
                    
                    # Adjust confidence for paper detection
                    if class_name == "Paper":
                        confidence = confidence * 5
                    
                    # Draw bounding box if confidence is high enough
                    if confidence > 0.2:
                        cv2.rectangle(
                            frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 
                            2
                        )
                        
                        # Draw detection label (what was detected)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(
                            frame, 
                            label, 
                            (int(x1), int(y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2
                        )
                        
                        # Draw response text (what to play to win)
                        if class_name == "Paper":
                            cv2.putText(
                                frame, 
                                "Play: SCISSORS!", 
                                (int(x1), int(y1 - 40)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, 
                                (0, 0, 255), 
                                2
                            )
                        elif class_name == "Rock":
                            cv2.putText(
                                frame, 
                                "Play: PAPER!", 
                                (int(x1), int(y1 - 40)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, 
                                (0, 0, 255), 
                                2
                            )
                        elif class_name == "Scissor":
                            cv2.putText(
                                frame, 
                                "Play: ROCK!", 
                                (int(x1), int(y1 - 40)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, 
                                (0, 0, 255), 
                                2
                            )
            
            # Add instruction text
            cv2.putText(
                frame, 
                "Show your hand gesture!", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        # Display frame
        cv2.imshow('Rock-Paper-Scissors YOLO Detector', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Rock-Paper-Scissors YOLO Detector stopped!")


if __name__ == '__main__':
    main()
