# Changelog

All notable changes to the Gesture Recognition project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the Gesture Recognition project
- Multiple gesture recognition approaches:
  - YOLO Classification for Rock-Paper-Scissors detection
  - YOLO Object Detection for hand gesture recognition
  - MediaPipe Hand Landmarks for real-time gesture analysis
- Real-time webcam detection for all approaches
- Training scripts for custom model development
- Comprehensive documentation and setup instructions
- Cross-platform support (Windows, macOS, Linux)

### Features
- **Classification Module**: YOLOv11n-cls based classification with 95% accuracy
- **Object Detection Module**: YOLOv11n based detection with bounding box visualization
- **MediaPipe Module**: Real-time hand landmark detection with vector-based gesture recognition
- **Training Pipeline**: Complete training scripts with data augmentation
- **Webcam Integration**: Real-time detection with OpenCV
- **Model Management**: Pre-trained models and weight management

### Technical Details
- **YOLO Classification**: 224x224 input, 10 epochs training, extensive data augmentation
- **YOLO Detection**: 640x640 input, 30 epochs training, comprehensive augmentation pipeline
- **MediaPipe**: 21 hand landmarks, vector-based recognition, 30+ FPS performance
- **Data Augmentation**: HSV variations, geometric transforms, advanced techniques (mosaic, mixup, copy-paste)

## [1.0.0] - 2024-01-XX

### Added
- Initial project structure with three main approaches
- README.md with comprehensive documentation
- requirements.txt with all necessary dependencies
- .gitignore for Python and ML projects
- MIT License
- setup.py for easy installation
- CONTRIBUTING.md for contributor guidelines
- CHANGELOG.md for version tracking

### Classification Module
- `train_classification_rps.py`: Training script for YOLO classification
- `rps_classification_webcam.py`: Real-time classification detection
- `test_classification_model.py`: Model testing and validation
- Pre-trained model weights in `rock_paper_scissors_robust/weights/`

### Object Recognition Module
- `train_detection_rps.py`: Training script for YOLO object detection
- `rps_detection_webcam.py`: Real-time object detection with bounding boxes
- `downloadDataset.py`: Kaggle dataset download utility
- Pre-trained model weights in `rps_yolo_training/rock_paper_scissors_detection/weights/`

### MediaPipe Module
- `rps_mediapipe_landmarks.py`: Real-time hand landmark detection
- Vector-based gesture recognition algorithm
- Hand landmarker model file

### Documentation
- Comprehensive README with installation and usage instructions
- Performance comparison table
- Troubleshooting guide
- Technical implementation details
- Future enhancement roadmap

### Dependencies
- ultralytics>=8.0.0
- opencv-python>=4.8.0
- mediapipe>=0.10.0
- torch>=2.0.0
- numpy>=1.24.0
- And other supporting libraries

---

## Future Releases

### Planned Features
- [ ] Support for additional gesture types (thumbs up, peace sign, etc.)
- [ ] Mobile app integration
- [ ] Real-time multiplayer game mode
- [ ] Improved accuracy with ensemble methods
- [ ] Web-based interface
- [ ] Custom gesture training interface
- [ ] Model quantization for mobile deployment
- [ ] Multi-hand detection support
- [ ] Gesture sequence recognition
- [ ] Performance optimization for edge devices

### Potential Improvements
- [ ] Better handling of lighting variations
- [ ] Improved accuracy for side-view gestures
- [ ] Real-time model fine-tuning
- [ ] Gesture confidence scoring
- [ ] Multi-language support
- [ ] Accessibility features
- [ ] Cloud deployment options
- [ ] API endpoints for integration

---

## Version History

- **v1.0.0**: Initial release with three gesture recognition approaches
- **v0.1.0**: Development version (internal)

---

## Notes

- All versions maintain backward compatibility unless noted
- Breaking changes will be clearly documented
- Migration guides will be provided for major version updates
- Performance benchmarks are available in the README
