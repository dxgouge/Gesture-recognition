# rps-gesture-recognition

Real-time Rock-Paper-Scissors gesture recognition built from scratch in C++ and Python. The system captures webcam video, detects hands, extracts skeletal landmarks, and classifies gestures — all running natively without Python or MediaPipe at runtime.

> **Built as a learning project** to go deep on C++, low-level ML inference, and computer vision — replacing a working Python/MediaPipe prototype with a fully native implementation piece by piece.



## How It Works

The pipeline runs in real time on a standard webcam feed:

```
Camera frame
  → Palm Detection (TFLite, BlazePalm)
  → Hand Landmark Estimation (TFLite, 21-point skeleton)
  → Feature Extraction (angles, distances, direction vectors)
  → Gesture Classification (LightGBM, 5-frame sliding window)
  → "Rock" / "Paper" / "Scissors"
```

Each stage was implemented from the ground up by studying MediaPipe's published model configs (`.pbtxt` files) and reimplementing the math — anchor generation, affine crop-and-rotate transforms, letterbox preprocessing, coordinate remapping — rather than wrapping an existing framework.

## Architecture

```
main.cpp                 Entry point, model loading
├── palm_detector.*      BlazePalm SSD detection + anchor decoding + NMS
├── hand_landmarker.*    224×224 crop/rotate, TFLite landmark inference, inverse transform
├── features.*           25 rotation-invariant features per frame (angles, distances, vectors)
├── inference.*          LightGBM C API, 5-frame sliding window (125 features)
└── capture.*            OpenCV camera loop, overlay rendering, debug visualization
```

### Palm Detection

Runs the BlazePalm Lite model on a 192×192 letterboxed input. Generates 2016 SSD anchors across 4 feature map layers (strides 8, 16, 16, 16) matching MediaPipe's `SsdAnchorsCalculator` config exactly. Raw model outputs are anchor-relative offsets — the decoder applies anchor centers, removes letterbox padding, and extracts wrist/middle-finger keypoints to compute hand rotation. Non-maximum suppression filters overlapping detections.

### Hand Landmark Estimation

Crops and rotates the detected palm region into a 224×224 input using an affine transform derived from first principles. The crop is scaled 2.6× and shifted upward (`shift_y: -0.5`) to capture the full hand, following MediaPipe's `HandLandmarkSubgraph` parameters. After inference, landmarks are mapped back to original frame coordinates via the inverse affine transform.

### Feature Extraction

Computes 25 features per frame from the 21 landmark positions — the same features used by the Python training pipeline:

- **8 angles**: finger direction vectors measured relative to the palm baseline (rotation-invariant)
- **4 distances**: fingertip-to-wrist, scaled by palm size for hand-size invariance
- **1 aggregated distance**: mean pairwise distance across all landmarks
- **12 direction vector components**: 6 normalized unit vectors × 2 (x, y)

### Gesture Classification

A LightGBM model trained on labeled gesture data, loaded via the C API. Classification uses a sliding window of 5 consecutive frames (125 total features) to smooth predictions and reduce noise. The model outputs probabilities for Rock, Paper, and Scissors.

## Tech Stack

| Component | Technology |
|---|---|
| Language | C++17 |
| Camera & Display | OpenCV |
| Palm Detection | TensorFlow Lite v2.17.0 (built from source, XNNPACK) |
| Hand Landmarks | TensorFlow Lite v2.17.0 |
| Gesture Classifier | LightGBM (C API) |
| Build System | CMake |
| Platform | macOS ARM64 (Apple Silicon) |

TFLite was compiled from source on ARM64 Mac with SSE4.1 emulation patches and psimd support for XNNPACK compatibility.

## Building

### Prerequisites

- CMake 3.16+
- OpenCV 4.x (`brew install opencv`)
- TensorFlow Lite v2.17.0 (built from source — see notes below)
- LightGBM (`brew install lightgbm`)

### TFLite from Source

TFLite doesn't ship prebuilt static libraries for macOS ARM64 with XNNPACK. The build requires patching SSE4.1 intrinsics for ARM translation and linking against `libcpuinfo.a` and other XNNPACK dependencies. See the CMakeLists.txt for the full list of static libraries.

### Build & Run

```bash
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

./rps_cpp ../rps_lgbm_model.txt ../palm_detection_lite.tflite ../hand_landmark_lite.tflite
```

### Models

The project uses two TFLite models from MediaPipe's published model zoo:

- **palm_detection_lite.tflite** — BlazePalm, 192×192 input
- **hand_landmark_lite.tflite** — 21-point hand skeleton, 224×224 input

The LightGBM gesture classifier (`rps_lgbm_model.txt`) is trained on labeled CSV data collected through the built-in `collect` mode.

## Data Collection

The project includes a single-threaded data collection mode for gathering training data:

```bash
./rps_cpp collect <palm_model> <landmark_model> <output.csv> <gesture_label>
```

This captures landmark coordinates and computed features to CSV, with columns matched exactly to the Python training script's expected format. Collection runs single-threaded (no frame dropping) to avoid gaps in the training data.

## What I Learned

This project was an exercise in going from "it works in Python" to understanding every layer of the stack:

- **SSD anchor mechanics**: raw model outputs are offsets relative to anchor grid positions, not absolute coordinates — skipping anchor decoding produces garbage
- **Affine transforms from first principles**: deriving the crop-rotate-scale matrix by hand rather than patching OpenCV's `getRotationMatrix2D`, and empirically validating the math
- **MediaPipe's config files are the source of truth**: `.pbtxt` files specify tensor ordering, ROI parameters, normalization ranges, anchor configs — reverse-engineering without these would be guesswork
- **Training/inference feature parity**: any difference in preprocessing between data collection and inference silently degrades the classifier
- **macOS threading constraints**: `cv::imshow` must run on the main thread or `NSWindow` crashes — threading decouples display from inference without violating this
- **Building C++ dependencies from source**: patching TFLite for ARM64, resolving XNNPACK/cpuinfo/psimd dependencies, managing static library linking in CMake

## License

MIT
