#include "palm_detector.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace rps {

PalmDetector::PalmDetector(const std::string& model_path) {
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_) throw std::runtime_error("Failed to load palm detection model: " + model_path);

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, 2);

    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if (!interpreter_) throw std::runtime_error("Failed to create palm detector interpreter");

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Failed to allocate palm detector tensors");
}

PalmDetector::~PalmDetector() {
    if (interpreter_) TfLiteInterpreterDelete(interpreter_);
    if (options_)     TfLiteInterpreterOptionsDelete(options_);
    if (model_)       TfLiteModelDelete(model_);
}

void PalmDetector::preprocess(const cv::Mat& frame, cv::Mat& out) {
    // Resize to 192x192
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));

    // Convert BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Normalize from [0,255] to [-1, 1] as float32
    rgb.convertTo(out, CV_32FC3, 1.0f / 127.5f, -1.0f);
}

std::vector<PalmDetection> PalmDetector::detect(const cv::Mat& frame) {
    // Preprocess
    cv::Mat input_mat;
    preprocess(frame, input_mat);

    // Copy into input tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    const int num_elements = INPUT_SIZE * INPUT_SIZE * 3;
    memcpy(TfLiteTensorData(input_tensor),
           input_mat.ptr<float>(0),
           num_elements * sizeof(float));

    // Run inference
    if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Palm detector inference failed");

    // Parse and return detections
    auto dets = parseOutput(frame.cols, frame.rows);
    return nms(dets, NMS_THRESH);
}

std::vector<PalmDetection> PalmDetector::parseOutput(int frame_w, int frame_h) {
    // Palm detection model outputs two tensors:
    // - output[0]: bounding box regressors [1, num_anchors, 18]
    // - output[1]: classification scores   [1, num_anchors, 1]
    const TfLiteTensor* boxes_tensor  = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    const TfLiteTensor* scores_tensor = TfLiteInterpreterGetOutputTensor(interpreter_, 1);

    const float* boxes  = reinterpret_cast<const float*>(TfLiteTensorData(boxes_tensor));
    const float* scores = reinterpret_cast<const float*>(TfLiteTensorData(scores_tensor));

    const int num_anchors = TfLiteTensorDim(boxes_tensor, 1);

    std::vector<PalmDetection> dets;

    for (int i = 0; i < num_anchors; ++i) {
        // Sigmoid to convert logit to probability
        float score = 1.0f / (1.0f + std::exp(-scores[i]));
        if (score < SCORE_THRESH) continue;

        // Box encoding: [cx, cy, w, h, kp0x, kp0y, ... kp6x, kp6y]
        // All values normalized to INPUT_SIZE
        const float* box = boxes + i * 18;
        float cx = box[0] / INPUT_SIZE;
        float cy = box[1] / INPUT_SIZE;
        float w  = box[2] / INPUT_SIZE;
        float h  = box[3] / INPUT_SIZE;

        // Compute rotation from keypoints 0 (wrist) and 2 (middle finger base)
        float kp0x = box[4]  / INPUT_SIZE;
        float kp0y = box[5]  / INPUT_SIZE;
        float kp2x = box[8]  / INPUT_SIZE;
        float kp2y = box[9]  / INPUT_SIZE;
        float rotation = std::atan2(kp0y - kp2y, kp0x - kp2x) - static_cast<float>(M_PI / 2.0);

        dets.push_back({cx, cy, w, h, rotation, score});
    }

    return dets;
}

std::vector<PalmDetection> PalmDetector::nms(std::vector<PalmDetection>& dets, float iou_thresh) {
    // Sort by score descending
    std::sort(dets.begin(), dets.end(),
              [](const PalmDetection& a, const PalmDetection& b) {
                  return a.score > b.score;
              });

    std::vector<PalmDetection> kept;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;

            // Compute IoU between dets[i] and dets[j]
            float ax1 = dets[i].cx - dets[i].width  / 2;
            float ay1 = dets[i].cy - dets[i].height / 2;
            float ax2 = dets[i].cx + dets[i].width  / 2;
            float ay2 = dets[i].cy + dets[i].height / 2;

            float bx1 = dets[j].cx - dets[j].width  / 2;
            float by1 = dets[j].cy - dets[j].height / 2;
            float bx2 = dets[j].cx + dets[j].width  / 2;
            float by2 = dets[j].cy + dets[j].height / 2;

            float ix1 = std::max(ax1, bx1);
            float iy1 = std::max(ay1, by1);
            float ix2 = std::min(ax2, bx2);
            float iy2 = std::min(ay2, by2);

            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
            float area_a = dets[i].width * dets[i].height;
            float area_b = dets[j].width * dets[j].height;
            float iou = inter / (area_a + area_b - inter);

            if (iou > iou_thresh) suppressed[j] = true;
        }
    }

    return kept;
}

} // namespace rps