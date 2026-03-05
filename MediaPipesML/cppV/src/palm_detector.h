#pragma once
#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/c/c_api.h"

namespace rps {

struct PalmDetection {
    float cx, cy;
    float width, height;
    float rotation;
    float score;
};

class PalmDetector {
public:
    struct Anchor { float cx, cy; };

    explicit PalmDetector(const std::string& model_path);
    ~PalmDetector();

    std::vector<PalmDetection> detect(const cv::Mat& frame);

private:
    float pad_x_ = 0.0f;   // normalized padding on each side (left/right)
    float pad_y_ = 0.0f;   // normalized padding on each side (top/bottom)
    float scale_ = 1.0f;   // scale factor used during letterbox resize
    void preprocess(const cv::Mat& frame, cv::Mat& out);
    std::vector<PalmDetection> parseOutput(int frame_w, int frame_h);
    std::vector<PalmDetection> nms(std::vector<PalmDetection>& dets, float iou_thresh);
    static std::vector<Anchor> generateAnchors();

    std::vector<Anchor>        anchors_;
    TfLiteModel*               model_       = nullptr;
    TfLiteInterpreter*         interpreter_ = nullptr;
    TfLiteInterpreterOptions*  options_     = nullptr;

    static constexpr int   INPUT_SIZE   = 192;
    static constexpr float SCORE_THRESH = 0.2f;
    static constexpr float NMS_THRESH   = 0.3f;
};

} // namespace rps
