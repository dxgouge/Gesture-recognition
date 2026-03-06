#include "palm_detector.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

namespace rps {

// Exact MediaPipe SsdAnchorsCalculator config for palm_detection_lite:
// num_layers: 4, strides: [8, 16, 16, 16]
// min_scale: 0.1484375, max_scale: 0.75
// input: 192x192, anchor_offset: 0.5, fixed_anchor_size: true
// aspect_ratios: [1.0], interpolated_scale_aspect_ratio: 1.0
// Produces 2016 anchors total.
std::vector<PalmDetector::Anchor> PalmDetector::generateAnchors() {
    const int   strides[]   = {8, 16, 16, 16};
    const int   num_layers  = 4;
    const float min_scale   = 0.1484375f;
    const float max_scale   = 0.75f;
    const int   input_w     = INPUT_SIZE;
    const int   input_h     = INPUT_SIZE;

    auto calculate_scale = [&](int layer) {
        if (num_layers == 1)
            return (min_scale + max_scale) * 0.5f;
        return min_scale + (max_scale - min_scale) * layer / (num_layers - 1.0f);
    };

    std::vector<Anchor> anchors;
    anchors.reserve(2016);

    for (int layer = 0; layer < num_layers; ++layer) {
        int stride = strides[layer];
        int feat_w = (int)std::ceil((float)input_w / stride);
        int feat_h = (int)std::ceil((float)input_h / stride);

        float scale      = calculate_scale(layer);
        float scale_next = (layer == num_layers - 1)
                           ? 1.0f
                           : calculate_scale(layer + 1);

        // Two anchors per cell: one at current scale, one at interpolated scale
        // (geometric mean of current and next scale)
        float interp = std::sqrt(scale * scale_next);

        for (int y = 0; y < feat_h; ++y) {
            for (int x = 0; x < feat_w; ++x) {
                float cx = (x + 0.5f) * stride / input_w;
                float cy = (y + 0.5f) * stride / input_h;
                anchors.push_back({cx, cy});  // anchor at current scale
                anchors.push_back({cx, cy});  // anchor at interpolated scale
                // Note: both anchors share the same center; the model's
                // regressor output encodes the actual w/h offsets per anchor.
            }
        }
    }

    return anchors;
}

PalmDetector::PalmDetector(const std::string& model_path) {
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("Failed to load palm detection model: " + model_path);

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, 4);
    

    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if (!interpreter_)
        throw std::runtime_error("Failed to create palm detector interpreter");

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Failed to allocate palm detector tensors");

    

    anchors_ = generateAnchors();
    std::cout << "PalmDetector: " << anchors_.size() << " anchors generated" << std::endl;
}

PalmDetector::~PalmDetector() {
    if (interpreter_) TfLiteInterpreterDelete(interpreter_);
    if (options_)     TfLiteInterpreterOptionsDelete(options_);
    if (model_)       TfLiteModelDelete(model_);
}

void PalmDetector::preprocess(const cv::Mat& frame, cv::Mat& out) {
    int fw = frame.cols, fh = frame.rows;
    float scale = std::min((float)INPUT_SIZE / fw, (float)INPUT_SIZE / fh);
    int new_w = (int)(fw * scale);
    int new_h = (int)(fh * scale);
    int pad_x = (INPUT_SIZE - new_w) / 2;
    int pad_y = (INPUT_SIZE - new_h) / 2;

    // Store normalized padding for decode step
    pad_x_ = (float)pad_x / INPUT_SIZE;
    pad_y_ = (float)pad_y / INPUT_SIZE;
    scale_ = scale;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));
    cv::Mat padded = cv::Mat::zeros(INPUT_SIZE, INPUT_SIZE, CV_8UC3);
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(out, CV_32FC3, 1.0f / 255.0f);
}

std::vector<PalmDetection> PalmDetector::detect(const cv::Mat& frame) {
    cv::Mat input_mat;
    preprocess(frame, input_mat);

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    const int num_elements = INPUT_SIZE * INPUT_SIZE * 3;
    memcpy(TfLiteTensorData(input_tensor),
           input_mat.ptr<float>(0),
           num_elements * sizeof(float));

    auto t0 = std::chrono::high_resolution_clock::now();
    if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Palm detector inference failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Palm: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0 
            << "ms" << std::endl;

    auto dets = parseOutput(frame.cols, frame.rows);
    return nms(dets, NMS_THRESH);
}

std::vector<PalmDetection> PalmDetector::parseOutput(int /*frame_w*/, int /*frame_h*/) {
    const TfLiteTensor* boxes_tensor  = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    const TfLiteTensor* scores_tensor = TfLiteInterpreterGetOutputTensor(interpreter_, 1);

    const float* boxes  = reinterpret_cast<const float*>(TfLiteTensorData(boxes_tensor));
    const float* scores = reinterpret_cast<const float*>(TfLiteTensorData(scores_tensor));

    const int num_anchors = static_cast<int>(anchors_.size());  // 2016
    std::vector<PalmDetection> dets;

    for (int i = 0; i < num_anchors; ++i) {
        // Raw score is a logit — apply sigmoid to get probability
        float score = 1.0f / (1.0f + std::exp(-scores[i]));
        if (score < SCORE_THRESH) continue;

        // Each anchor has 18 box values:
        //   [0,1]   = cx, cy offset (in input pixels, relative to anchor center)
        //   [2,3]   = w, h          (in input pixels)
        //   [4..17] = 7 keypoints x (x, y) each
        const float* box = boxes + i * 18;

        // Decode: anchor center + offset, divide by input size to normalize to [0,1]
        float cx = anchors_[i].cx + box[0] / INPUT_SIZE;
        float cy = anchors_[i].cy + box[1] / INPUT_SIZE;
        float w  = box[2] / INPUT_SIZE;
        float h  = box[3] / INPUT_SIZE;


        
        // Keypoint 0 = wrist, keypoint 2 = middle finger MCP
        // Rotation is the angle of the wrist -> middle-MCP vector, offset by -90°
        // so that a vertical hand = 0 rotation.
        float kp0x = anchors_[i].cx + box[4] / INPUT_SIZE;
        float kp0y = anchors_[i].cy + box[5] / INPUT_SIZE;
        float kp2x = anchors_[i].cx + box[8] / INPUT_SIZE;
        float kp2y = anchors_[i].cy + box[9] / INPUT_SIZE;

        // After decoding cx, cy, w, h from anchor offsets:

        // Remove letterbox padding — map from letterboxed [0,1] to original frame [0,1]
        cx = (cx - pad_x_) / (1.0f - 2.0f * pad_x_);
        cy = (cy - pad_y_) / (1.0f - 2.0f * pad_y_);
        w  =  w            / (1.0f - 2.0f * pad_x_);
        h  =  h            / (1.0f - 2.0f * pad_y_);

        // Same for keypoints used for rotation
        kp0x = (kp0x - pad_x_) / (1.0f - 2.0f * pad_x_);
        kp0y = (kp0y - pad_y_) / (1.0f - 2.0f * pad_y_);
        kp2x = (kp2x - pad_x_) / (1.0f - 2.0f * pad_x_);
        kp2y = (kp2y - pad_y_) / (1.0f - 2.0f * pad_y_);
        
        // was: kp0 - kp2 (backwards)


        //float rotation = std::atan2(kp2y - kp0y, kp2x - kp0x) - static_cast<float>(M_PI * 0.5f);
        float rotation = std::atan2(kp0y - kp2y, kp0x - kp2x) - M_PI / 2.0;

        dets.push_back({cx, cy, w, h, rotation, score});


    }
    // Print first 5 raw scores (before sigmoid) to see what the model is outputting
    // std::cout << "Raw scores[0..4]: ";
    // for (int i = 0; i < 5; ++i)
    //     std::cout << scores[i] << " ";
    // std::cout << std::flush;

    // Count how many pass threshold
    int count = 0;
    for (int i = 0; i < 2016; ++i) {
        float s = 1.0f / (1.0f + std::exp(-scores[i]));
        if (s > SCORE_THRESH) count++;
    }
    //std::cout << "Detections above threshold: " << count << std::flush;

    return dets;
}

std::vector<PalmDetection> PalmDetector::nms(std::vector<PalmDetection>& dets, float iou_thresh) {
    // Sort highest score first
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

            float ax1 = dets[i].cx - dets[i].width  / 2;
            float ay1 = dets[i].cy - dets[i].height / 2;
            float ax2 = dets[i].cx + dets[i].width  / 2;
            float ay2 = dets[i].cy + dets[i].height / 2;

            float bx1 = dets[j].cx - dets[j].width  / 2;
            float by1 = dets[j].cy - dets[j].height / 2;
            float bx2 = dets[j].cx + dets[j].width  / 2;
            float by2 = dets[j].cy + dets[j].height / 2;

            float ix1  = std::max(ax1, bx1);
            float iy1  = std::max(ay1, by1);
            float ix2  = std::min(ax2, bx2);
            float iy2  = std::min(ay2, by2);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);

            float area_a = dets[i].width * dets[i].height;
            float area_b = dets[j].width * dets[j].height;
            float iou    = inter / (area_a + area_b - inter);

            if (iou > iou_thresh) suppressed[j] = true;
        }
    }

    return kept;
}

} // namespace rps