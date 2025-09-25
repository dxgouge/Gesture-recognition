from ultralytics import YOLO
import os


def create_dataset_config():
    """Create dataset configuration for Rock-Paper-Scissors classification."""
    dataset_config = """
# Rock-Paper-Scissors Dataset Configuration
path: ./archive (1)/Rock-Paper-Scissors/Rock-Paper-Scissors  # dataset root dir
train: train  # train images (relative to 'path')
val: validation  # val images (relative to 'path')

# Classes
names:
  0: paper
  1: rock
  2: scissors
"""

    # Write dataset.yaml file
    yaml_path = os.path.join(os.path.dirname(__file__), 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(dataset_config)

    print(f"Dataset configuration created at: {yaml_path}")
    return yaml_path

def train_model():
    """Train the YOLO model for rock-paper-scissors classification."""
    # Load a pre-trained YOLO model for classification
    print("Loading YOLO model...")
    model = YOLO('yolo11n-cls.pt')  # Use classification model
    print("Model loaded successfully!")

    # Train the model
    print("Starting training...")
    results = model.train(
        data=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive (1)', 'Rock-Paper-Scissors', 'Rock-Paper-Scissors'),
        epochs=10,  # Number of training epochs
        imgsz=224,  # Image size
        batch=8,    # Reduced batch size for stability
        device='cuda',  # Use GPU for faster training
        project=os.path.join(os.path.dirname(__file__), 'rps_training'),
        name='rock_paper_scissors_robust',
        val=True,   # Enable validation
        patience=10,  # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        workers=4,   # Reduced workers for stability
        # Data augmentation for robustness
        hsv_h=0.015,  # Hue variation
        hsv_s=0.7,    # Saturation variation
        hsv_v=0.4,    # Value/brightness variation
        degrees=15,   # Rotation variation
        translate=0.1, # Translation variation
        scale=0.5,    # Scale variation
        shear=2.0,    # Shear variation
        perspective=0.0, # Perspective variation
        flipud=0.0,   # Vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        copy_paste=0.1, # Copy-paste augmentation
        erasing=0.4,  # Random erasing
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
    test_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive (1)', 'Rock-Paper-Scissors', 'Rock-Paper-Scissors', 'validation', 'rock', 'rock1.png')
    test_results = trained_model.predict(source=test_image, show=True)
    print("Test completed!")


def main():
    """Main function to train and test the YOLO classification model."""
    # Create dataset configuration
    create_dataset_config()
    
    # Train the model
    results = train_model()
    
    # Test the model
    test_model(results)


if __name__ == '__main__':
    main()
