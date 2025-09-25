from ultralytics import YOLO
import os


def test_model(model_path, test_image):
    """Test the trained classification model on a sample image."""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False
    
    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return False
    
    print(f"Testing on: {test_image}")
    results = model.predict(source=test_image, show=True)
    
    # Print the prediction
    for result in results:
        predicted_class = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
    
    return True


def main():
    """Main function to test the classification model."""
    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'rock_paper_scissors_robust', 'weights', 'best.pt')
    
    # Test on a sample image
    test_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive (1)', 'Rock-Paper-Scissors', 'Rock-Paper-Scissors', 'validation', 'rock', 'rock1.png')
    
    test_model(model_path, test_image)


if __name__ == '__main__':
    main()
