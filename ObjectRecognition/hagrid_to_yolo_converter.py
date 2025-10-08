import json
import os
import shutil
from pathlib import Path
import random
from typing import Dict, List, Tuple

def convert_hagrid_to_yolo(hagrid_root: str, output_root: str, train_split: float = 0.8, val_split: float = 0.1):
    """
    Convert HAGRID dataset to YOLO format.
    
    Args:
        hagrid_root: Path to HAGRID dataset root directory
        output_root: Path where YOLO dataset will be created
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
    """
    
    # Create output directories
    output_path = Path(output_root)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Define gesture classes (focusing on RPS + common gestures)
    gesture_classes = {
        "rock": 0,
        "palm": 1,  # Paper equivalent
        "peace": 2,  # Scissors equivalent
        "fist": 3,
        "ok": 4,
        "like": 5,
        "dislike": 6,
        "stop": 7,
        "call": 8,
        "mute": 9,
        "one": 10,
        "two_up": 11,
        "three": 12,
        "four": 13
    }
    
    # Process each gesture
    all_images = []
    
    for gesture_name, class_id in gesture_classes.items():
        gesture_dir = Path(hagrid_root) / "hagrid_30k" / f"train_val_{gesture_name}"
        annotation_file = Path(hagrid_root) / "ann_train_val" / f"{gesture_name}.json"
        
        if not gesture_dir.exists() or not annotation_file.exists():
            print(f"Warning: {gesture_name} directory or annotation file not found, skipping...")
            continue
            
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Process each image
        for image_file in gesture_dir.glob("*.jpg"):
            image_id = image_file.stem
            
            if image_id in annotations:
                annotation = annotations[image_id]
                
                # Create YOLO label file
                label_file = labels_dir / "temp" / f"{image_id}.txt"
                label_file.parent.mkdir(exist_ok=True)
                
                with open(label_file, 'w') as f:
                    for i, bbox in enumerate(annotation.get('bboxes', [])):
                        label = annotation.get('labels', [])[i] if i < len(annotation.get('labels', [])) else gesture_name
                        
                        # Convert to YOLO format (normalized center coordinates)
                        x_center, y_center, width, height = bbox
                        
                        # Write YOLO format: class_id x_center y_center width height
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                all_images.append({
                    'image_path': image_file,
                    'label_path': label_file,
                    'gesture': gesture_name
                })
    
    # Split data into train/val/test
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    # Move files to appropriate directories
    for split, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        for item in images:
            # Copy image
            dest_image = images_dir / split / item['image_path'].name
            shutil.copy2(item['image_path'], dest_image)
            
            # Move label
            dest_label = labels_dir / split / f"{item['image_path'].stem}.txt"
            shutil.move(str(item['label_path']), str(dest_label))
    
    # Clean up temp directory
    temp_dir = labels_dir / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Create dataset.yaml
    dataset_yaml = f"""# HAGRID Dataset Configuration for YOLO
path: {output_root}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
names:
  0: rock
  1: palm
  2: peace
  3: fist
  4: ok
  5: like
  6: dislike
  7: stop
  8: call
  9: mute
  10: one
  11: two_up
  12: three
  13: four
"""
    
    yaml_path = output_path / "hagrid_dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)
    
    print(f"Dataset conversion completed!")
    print(f"Total images processed: {n_total}")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    print(f"Dataset YAML created at: {yaml_path}")
    
    return str(yaml_path)

if __name__ == "__main__":
    # Convert HAGRID dataset
    hagrid_root = "../../hagrid-sample-30k-384p"
    output_root = "./hagrid_yolo_dataset"
    
    yaml_path = convert_hagrid_to_yolo(hagrid_root, output_root)
    print(f"YOLO dataset ready at: {output_root}")


