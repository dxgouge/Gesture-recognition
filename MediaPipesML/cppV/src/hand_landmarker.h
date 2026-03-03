#pragma once
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/c/c_api.h"
#include "palm_detector.h"

namespace rps {

// 21 landmarks, each with x and y (normalized 0-1 in original frame space)
using Landmarks = std::array<float, 42>;

struct LandmarkResult {
    Landmarks landmarks;
    float     presence;   // confidence that a hand is present
    bool      valid;      // false if presence is too low
};

class HandLandmarker {
public:
    explicit HandLandmarker(const std::string& model_path);
    ~HandLandmarker();

    // Given the original frame and a palm detection, run landmark inference.
    // Returns 21 landmarks in original frame coordinate space.
    LandmarkResult detect(const cv::Mat& frame, const PalmDetection& palm);

private:
    // Crop and rotate the frame around the detected palm, resize to 224x224
    cv::Mat cropAndRotate(const cv::Mat& frame, const PalmDetection& palm,
                          cv::Mat& transform_matrix);

    // Map landmark coords from cropped 224x224 space back to original frame
    void transformLandmarks(Landmarks& landmarks,
                            const cv::Mat& transform_matrix,
                            int frame_w, int frame_h);

    TfLiteModel*       model_       = nullptr;
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteInterpreterOptions*     options_     = nullptr;

    static constexpr int   INPUT_SIZE     = 224;
    static constexpr float PRESENCE_THRESH = 0.5f;
    // Scale factor: how much larger than the detected palm to crop
    static constexpr float CROP_SCALE     = 2.6f;
};

} // namespace rps
