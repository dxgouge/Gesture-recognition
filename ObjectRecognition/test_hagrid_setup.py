#!/usr/bin/env python3
"""
Test script to verify HAGRID dataset setup and conversion.
Run this before training to ensure everything is configured correctly.
"""

import os
import sys
from pathlib import Path

def test_hagrid_dataset():
    """Test if HAGRID dataset is properly set up."""
    print("=== Testing HAGRID Dataset Setup ===")
    
    # Check if HAGRID dataset exists
    hagrid_root = Path("../../hagrid-sample-30k-384p")
    if not hagrid_root.exists():
        print("❌ HAGRID dataset not found at:", hagrid_root)
        print("Please ensure the dataset is downloaded and placed in the correct location.")
        return False
    
    print("✅ HAGRID dataset found at:", hagrid_root)
    
    # Check annotation files
    ann_dir = hagrid_root / "ann_train_val"
    if not ann_dir.exists():
        print("❌ Annotation directory not found:", ann_dir)
        return False
    
    print("✅ Annotation directory found:", ann_dir)
    
    # Check image directories
    images_dir = hagrid_root / "hagrid_30k"
    if not images_dir.exists():
        print("❌ Images directory not found:", images_dir)
        return False
    
    print("✅ Images directory found:", images_dir)
    
    # Check for key gesture directories
    key_gestures = ["rock", "palm", "peace", "fist", "ok"]
    for gesture in key_gestures:
        gesture_dir = images_dir / f"train_val_{gesture}"
        if gesture_dir.exists():
            image_count = len(list(gesture_dir.glob("*.jpg")))
            print(f"✅ {gesture}: {image_count} images")
        else:
            print(f"⚠️  {gesture}: directory not found")
    
    return True

def test_converter():
    """Test the HAGRID to YOLO converter."""
    print("\n=== Testing HAGRID to YOLO Converter ===")
    
    try:
        from hagrid_to_yolo_converter import convert_hagrid_to_yolo
        print("✅ Converter module imported successfully")
        
        # Test with a small subset (you can modify this for a full test)
        print("Note: Full conversion will be done during training.")
        return True
        
    except ImportError as e:
        print("❌ Failed to import converter:", e)
        return False
    except Exception as e:
        print("❌ Error in converter:", e)
        return False

def test_yolo_requirements():
    """Test if YOLO requirements are met."""
    print("\n=== Testing YOLO Requirements ===")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
        
        # Test model loading
        model = YOLO('yolo11n.pt')
        print("✅ YOLO11n model loaded successfully")
        
        return True
        
    except ImportError as e:
        print("❌ Failed to import ultralytics:", e)
        print("Please install: pip install ultralytics")
        return False
    except Exception as e:
        print("❌ Error loading YOLO model:", e)
        return False

def main():
    """Run all tests."""
    print("HAGRID Dataset Setup Test")
    print("=" * 50)
    
    tests = [
        test_hagrid_dataset,
        test_converter,
        test_yolo_requirements
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    if all(results):
        print("🎉 All tests passed! Ready to train HAGRID gesture detection model.")
        print("\nNext steps:")
        print("1. Run the training notebook: train_detection_rps.ipynb")
        print("2. The dataset will be automatically converted during training")
        print("3. Training will start with optimized parameters for gesture detection")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
        print("\nCommon solutions:")
        print("- Ensure HAGRID dataset is downloaded and in the correct location")
        print("- Install required packages: pip install ultralytics")
        print("- Check file permissions and paths")

if __name__ == "__main__":
    main()

