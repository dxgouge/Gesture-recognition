from ultralytics import YOLO
import cv2
import numpy as np
import os


def main():
    """Main function to run the Rock-Paper-Scissors classification detector."""
    # Load your trained model
    model_path = os.path.join(os.path.dirname(__file__), "rock_paper_scissors_robust", "weights", "best.pt")
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Rock-Paper-Scissors Detector Started!")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run prediction on the frame
        results = model.predict(frame, verbose=False)
        
        # Draw prediction on frame
        for result in results:
            # Get prediction
            class_id = result.probs.top1
            confidence = result.probs.top1conf
            class_name = result.names[class_id]
            
            # Draw text on frame
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                frame, 
                text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Add instruction text
            cv2.putText(
                frame, 
                "Show your hand gesture!", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        # Display frame
        cv2.imshow('Rock-Paper-Scissors Detector', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Rock-Paper-Scissors Detector stopped!")


if __name__ == '__main__':
    main()
