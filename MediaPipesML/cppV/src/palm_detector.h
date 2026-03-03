#pragma once
#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/c/c_api.h"

namespace rps {

// A detected palm in the original frame's coordinate space
struct PalmDetection {
    float cx, cy;        // center x, y (normalized 0-1)
    float width, height; // bounding box size (normalized 0-1)
    float rotation;      // hand rotation angle in radians
    float score;         // detection confidence
};

class PalmDetector {
public:
    explicit PalmDetector(const std::string& model_path);
    ~PalmDetector();

    // Run palm detection on a BGR frame.
    // Returns list of detected palms (usually 0 or 1).
    std::vector<PalmDetection> detect(const cv::Mat& frame);

private:
    // Preprocess: resize to 192x192, normalize to [-1, 1]
    void preprocess(const cv::Mat& frame, cv::Mat& out);

    // Parse raw model output tensors into PalmDetection structs
    std::vector<PalmDetection> parseOutput(int frame_w, int frame_h);

    // Non-maximum suppression to remove duplicate detections
    std::vector<PalmDetection> nms(std::vector<PalmDetection>& dets, float iou_thresh);

    TfLiteModel*       model_       = nullptr;
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteInterpreterOptions*     options_     = nullptr;

    static constexpr int   INPUT_SIZE  = 192;
    static constexpr float SCORE_THRESH = 0.5f;
    static constexpr float NMS_THRESH   = 0.3f;
};

} // namespace rps