#include "hand_landmarker.h"
#include <stdexcept>
#include <cmath>

namespace rps {

HandLandmarker::HandLandmarker(const std::string& model_path) {
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_) throw std::runtime_error("Failed to load hand landmark model: " + model_path);

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, 2);

    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if (!interpreter_) throw std::runtime_error("Failed to create hand landmarker interpreter");

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Failed to allocate hand landmarker tensors");
}

HandLandmarker::~HandLandmarker() {
    if (interpreter_) TfLiteInterpreterDelete(interpreter_);
    if (options_)     TfLiteInterpreterOptionsDelete(options_);
    if (model_)       TfLiteModelDelete(model_);
}

cv::Mat HandLandmarker::cropAndRotate(const cv::Mat& frame,
                                       const PalmDetection& palm,
                                       cv::Mat& transform_matrix) {
    float frame_w = static_cast<float>(frame.cols);
    float frame_h = static_cast<float>(frame.rows);

    // Convert normalized palm center to pixel coords
    float cx = palm.cx * frame_w;
    float cy = palm.cy * frame_h;

    // Crop size: take the larger dimension of the palm box, scaled up
    float box_size = std::max(palm.width * frame_w, palm.height * frame_h) * CROP_SCALE;
    float half     = box_size / 2.0f;

    // Build the transformation matrix:
    // We want to rotate around the palm center and crop to a square.
    // OpenCV's getRotationMatrix2D does: rotate around center, then translate.
    // We then map that rotated square into a 224x224 output image.

    // Step 1: rotate around palm center to align hand upright
    float angle_deg = palm.rotation * 180.0f / static_cast<float>(M_PI);
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(cx, cy), angle_deg, 1.0f);

    // Step 2: after rotation, our crop square is centered at (cx, cy)
    // with half-size = half. We want to map:
    //   top-left (cx-half, cy-half) -> (0, 0)
    //   bottom-right (cx+half, cy+half) -> (224, 224)
    // Add translation to rot matrix to achieve this
    rot.at<double>(0, 2) += (INPUT_SIZE / 2.0 - cx);
    rot.at<double>(1, 2) += (INPUT_SIZE / 2.0 - cy);

    // Scale: the crop square of size box_size must fit into INPUT_SIZE pixels
    double scale = INPUT_SIZE / box_size;
    rot.at<double>(0, 0) *= scale; rot.at<double>(0, 1) *= scale; rot.at<double>(0, 2) *= scale;
    rot.at<double>(1, 0) *= scale; rot.at<double>(1, 1) *= scale; rot.at<double>(1, 2) *= scale;
    // Re-apply the non-scaled translation component
    rot.at<double>(0, 2) += (1.0 - scale) * (INPUT_SIZE / 2.0);
    rot.at<double>(1, 2) += (1.0 - scale) * (INPUT_SIZE / 2.0);

    transform_matrix = rot.clone();

    // Apply the transformation to produce a 224x224 cropped image
    cv::Mat cropped;
    cv::warpAffine(frame, cropped, rot, cv::Size(INPUT_SIZE, INPUT_SIZE),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // Convert BGR -> RGB, normalize [0,255] -> [0, 1]
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    cv::Mat out;
    rgb.convertTo(out, CV_32FC3, 1.0f / 255.0f);

    return out;
}

void HandLandmarker::transformLandmarks(Landmarks& landmarks,
                                         const cv::Mat& transform_matrix,
                                         int frame_w, int frame_h) {
    // Invert the transform matrix so we can map 224x224 coords -> original frame
    cv::Mat inv;
    cv::invertAffineTransform(transform_matrix, inv);

    for (int i = 0; i < 21; ++i) {
        float x = landmarks[i * 2    ] * INPUT_SIZE;
        float y = landmarks[i * 2 + 1] * INPUT_SIZE;

        // Apply inverse affine transform
        float ox = static_cast<float>(inv.at<double>(0,0) * x +
                                      inv.at<double>(0,1) * y +
                                      inv.at<double>(0,2));
        float oy = static_cast<float>(inv.at<double>(1,0) * x +
                                      inv.at<double>(1,1) * y +
                                      inv.at<double>(1,2));

        // Normalize back to 0-1
        landmarks[i * 2    ] = ox / frame_w;
        landmarks[i * 2 + 1] = oy / frame_h;
    }
}

LandmarkResult HandLandmarker::detect(const cv::Mat& frame, const PalmDetection& palm) {
    // Crop and rotate around the detected palm
    cv::Mat transform_matrix;
    cv::Mat input_mat = cropAndRotate(frame, palm, transform_matrix);

    // Copy into input tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    const int num_elements = INPUT_SIZE * INPUT_SIZE * 3;
    memcpy(TfLiteTensorData(input_tensor),
           input_mat.ptr<float>(0),
           num_elements * sizeof(float));

    // Run inference
    if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Hand landmark inference failed");

    // Output tensors:
    // [0]: landmarks  [1, 63]  - 21 landmarks x (x, y, z)
    // [1]: presence   [1, 1]   - hand presence score (sigmoid)
    // [2]: handedness [1, 1]   - left/right hand score
    const TfLiteTensor* lm_tensor  = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    const TfLiteTensor* pr_tensor  = TfLiteInterpreterGetOutputTensor(interpreter_, 1);

    const float* lm_data = reinterpret_cast<const float*>(TfLiteTensorData(lm_tensor));
    const float* pr_data = reinterpret_cast<const float*>(TfLiteTensorData(pr_tensor));

    float presence = 1.0f / (1.0f + std::exp(-pr_data[0]));

    if (presence < PRESENCE_THRESH) {
        return {{}, presence, false};
    }

    // Extract x, y only (skip z) and normalize to [0, 1] in cropped space
    Landmarks landmarks;
    for (int i = 0; i < 21; ++i) {
        landmarks[i * 2    ] = lm_data[i * 3    ] / INPUT_SIZE;
        landmarks[i * 2 + 1] = lm_data[i * 3 + 1] / INPUT_SIZE;
    }

    // Map back to original frame coordinates
    transformLandmarks(landmarks, transform_matrix, frame.cols, frame.rows);

    return {landmarks, presence, true};
}

} // namespace rps
