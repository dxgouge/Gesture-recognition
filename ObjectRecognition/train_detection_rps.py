from ultralytics import YOLO
import os


def create_dataset_config():
    """Create dataset configuration for YOLO annotated dataset."""
    dataset_config = """
# Rock-Paper-Scissors YOLO Dataset Configuration
path: ./archive (1)/RPS_YOLO_Annotated  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
names:
  0: Rock
  1: Paper
  2: Scissor
"""

    # Write dataset.yaml file
    yaml_path = os.path.join(os.path.dirname(__file__), 'rps_yolo_dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(dataset_config)

    print(f"Dataset configuration created at: {yaml_path}")
    return yaml_path


def train_model(yaml_path):
    """Train the YOLO model for rock-paper-scissors detection."""
    # Load a pre-trained YOLO model for object detection
    print("Loading YOLO model...")
    model = YOLO('yolo11n.pt')  # Use detection model (not classification)
    print("Model loaded successfully!")

    # Train the model
    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=30,  # Number of training epochs
        imgsz=640,  # Image size for detection
        batch=8,    # Batch size
        device='cuda',  # Use GPU for faster training
        project=os.path.join(os.path.dirname(__file__), 'rps_yolo_training'),
        name='rock_paper_scissors_detection',
        val=True,   # Enable validation
        patience=10,  # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        workers=4,   # Reduced workers for stability
        # Data augmentation for robustness
        hsv_h=0.015,  # Hue variation
        hsv_s=0.7,    # Saturation variation
        hsv_v=0.4,    # Value/brightness variation
        degrees=10,   # Rotation variation
        translate=0.1, # Translation variation
        scale=0.5,    # Scale variation
        shear=2.0,    # Shear variation
        perspective=0.0, # Perspective variation
        flipud=0.0,   # Vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        copy_paste=0.1, # Copy-paste augmentation
        auto_augment='randaugment'  # Auto augmentation
    )

    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    return results

def test_model(results):
    """Test the trained model on a sample image."""
    # Test the trained model
    print("\nTesting the trained model...")
    trained_model = YOLO(f'{results.save_dir}/weights/best.pt')

    # Test on a sample image
    test_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive (1)', 'RPS_YOLO_Annotated', 'images', 'val', 'paper_back (1).jpg')
    if os.path.exists(test_image):
        test_results = trained_model.predict(source=test_image, show=True, save=True)
        print("Test completed!")
    else:
        print(f"Test image not found: {test_image}")


def main():
    """Main function to train and test the YOLO model."""
    # Create dataset configuration
    yaml_path = create_dataset_config()
    
    # Train the model
    results = train_model(yaml_path)
    
    # Test the model
    test_model(results)


if __name__ == '__main__':
    main()
