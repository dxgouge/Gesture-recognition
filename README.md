# Gesture Recognition Project

A comprehensive gesture recognition system that implements multiple approaches for detecting and classifying Rock-Paper-Scissors hand gestures using computer vision and machine learning.

## 🎯 Features

- **Multiple Recognition Methods**: 
  - YOLO Object Detection
  - YOLO Classification
  - MediaPipe Hand Landmarks
- **Real-time Webcam Detection**: Live gesture recognition with webcam input
- **Training Scripts**: Complete training pipelines for custom models
- **Pre-trained Models**: Ready-to-use models for immediate testing
- **Cross-platform Support**: Works on Windows, macOS, and Linux

## 📁 Project Structure

```
GestureRecognition/
├── Classification/              # YOLO Classification approach
│   ├── train_classification_rps.py
│   ├── rps_classification_webcam.py
│   ├── test_classification_model.py
│   ├── dataset.yaml
│   └── rock_paper_scissors_robust/
│       └── weights/
│           ├── best.pt
│           ├── last.pt
│           └── epoch0.pt
├── ObjectRecognition/           # YOLO Object Detection approach
│   ├── train_detection_rps.py
│   ├── rps_detection_webcam.py
│   ├── downloadDataset.py
│   ├── rps_yolo_dataset.yaml
│   └── rps_yolo_training/
│       └── rock_paper_scissors_detection/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── MediapipesGestureRecognition/ # MediaPipe Hand Landmarks approach
│   ├── rps_mediapipe_landmarks.py
│   └── hand_landmarker.task
└── archive (1)/                 # Dataset files
    ├── RPS_Raw_Images/
    ├── RPS_YOLO_Annotated/
    └── Rock-Paper-Scissors/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Webcam or camera device
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/gesture-recognition.git
   cd gesture-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets (optional):**
   ```bash
   cd ObjectRecognition
   python downloadDataset.py
   ```

### Usage

#### 1. YOLO Classification (Recommended for beginners)

```bash
cd Classification
python rps_classification_webcam.py
```

#### 2. YOLO Object Detection

```bash
cd ObjectRecognition
python rps_detection_webcam.py
```

#### 3. MediaPipe Hand Landmarks

```bash
cd MediapipesGestureRecognition
python rps_mediapipe_landmarks.py
```

## 🎮 How to Use

1. **Run any of the webcam scripts**
2. **Position your hand in front of the camera**
3. **Make Rock, Paper, or Scissors gestures**
4. **The system will detect and classify your gesture in real-time**
5. **Press 'q' to quit**

## 🏋️ Training Your Own Models

### Classification Model Training

```bash
cd Classification
python train_classification_rps.py
```

### Object Detection Model Training

```bash
cd ObjectRecognition
python train_detection_rps.py
```

## 📊 Model Performance

| Method | Accuracy | Speed | GPU Required |
|--------|----------|-------|--------------|
| YOLO Classification | ~95% | Fast | Optional |
| YOLO Detection | ~90% | Medium | Recommended |
| MediaPipe Landmarks | ~85% | Very Fast | No |

## 🛠️ Technical Details

### YOLO Classification
- **Model**: YOLOv11n-cls
- **Input Size**: 224x224
- **Classes**: Rock, Paper, Scissors
- **Training**: 10 epochs with data augmentation

### YOLO Object Detection
- **Model**: YOLOv11n
- **Input Size**: 640x640
- **Classes**: Rock, Paper, Scissors
- **Training**: 30 epochs with extensive augmentation

### MediaPipe Hand Landmarks
- **Framework**: MediaPipe Tasks API
- **Landmarks**: 21 hand keypoints
- **Method**: Vector-based gesture recognition
- **Real-time**: 30+ FPS

## 📈 Data Augmentation

The training scripts include comprehensive data augmentation:

- **HSV variations**: Hue, saturation, and value adjustments
- **Geometric transforms**: Rotation, translation, scaling, shearing
- **Advanced techniques**: Mosaic, mixup, copy-paste augmentation
- **Auto augmentation**: RandAugment for optimal performance

## 🔧 Configuration

### Dataset Configuration

Each approach uses YAML configuration files:

- `Classification/dataset.yaml` - Classification dataset paths
- `ObjectRecognition/rps_yolo_dataset.yaml` - Detection dataset paths

### Model Parameters

Key parameters can be adjusted in the training scripts:

- **Epochs**: Number of training iterations
- **Batch size**: Training batch size
- **Image size**: Input image dimensions
- **Learning rate**: Model learning rate
- **Data augmentation**: Augmentation parameters

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not detected:**
   - Ensure your camera is connected and not used by other applications
   - Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **CUDA out of memory:**
   - Reduce batch size in training scripts
   - Use CPU training by setting `device='cpu'`

3. **Model files not found:**
   - Ensure you've trained the models first
   - Check that the weights are in the correct directories

4. **Poor detection accuracy:**
   - Ensure good lighting conditions
   - Keep your hand clearly visible in the camera frame
   - Try different gesture positions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [Kaggle](https://www.kaggle.com/) for the Rock-Paper-Scissors dataset
- The computer vision community for inspiration and resources

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/gesture-recognition/issues) page
2. Create a new issue with detailed information
3. Include your system specifications and error messages

## 🔮 Future Enhancements

- [ ] Support for more gesture types
- [ ] Improved accuracy ML models

---


