#include "hand_landmarker.h"
#include <stdexcept>
#include <cmath>
#include <chrono>

namespace rps {

HandLandmarker::HandLandmarker(const std::string& model_path) {
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_) throw std::runtime_error("Failed to load hand landmark model: " + model_path);

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, 4);
    
    

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

    float cx = palm.cx * frame_w;
    float cy = palm.cy * frame_h;

    // Square crop using longer side, scaled by 2.6
    float box_size = std::max(palm.width * frame_w, palm.height * frame_h) * CROP_SCALE;
    
    // shift_y: -0.5 means shift center upward by half the box size
    // This moves the crop center from palm to middle of full hand
    float shift_y = -0.5f * box_size * 0.5f ;
    float cos_r   = std::cos(palm.rotation);
    float sin_r   = std::sin(palm.rotation);

    // Apply shift in the rotated frame's Y direction
    cx += -sin_r * shift_y;
    cy +=  cos_r * shift_y;

    double scale = static_cast<double>(INPUT_SIZE) / box_size;
    double cos_t = std::cos(static_cast<double>(-palm.rotation)) * scale;
    double sin_t = std::sin(static_cast<double>(-palm.rotation)) * scale;

    double half = INPUT_SIZE / 2.0;
    double tx = -cos_t * cx + sin_t * cy + half;
    double ty = -sin_t * cx - cos_t * cy + half;

    transform_matrix = cv::Mat(2, 3, CV_64F);
    transform_matrix.at<double>(0, 0) =  cos_t;
    transform_matrix.at<double>(0, 1) = -sin_t;
    transform_matrix.at<double>(0, 2) =  tx;
    transform_matrix.at<double>(1, 0) =  sin_t;
    transform_matrix.at<double>(1, 1) =  cos_t;
    transform_matrix.at<double>(1, 2) =  ty;

    cv::Mat cropped;
    cv::warpAffine(frame, cropped, transform_matrix,
                   cv::Size(INPUT_SIZE, INPUT_SIZE),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    cv::Mat out;
    rgb.convertTo(out, CV_32FC3, 1.0f / 255.0f);
    cv::Mat display;
cv::cvtColor(cropped, display, cv::COLOR_BGR2RGB);
cv::imshow("Landmarker Input", display);
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
    // std::cout << "rotation=" << palm.rotation 
    //       << " (" << palm.rotation * 180.0f / M_PI << " deg)" << std::flush;
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    const int num_elements = INPUT_SIZE * INPUT_SIZE * 3;
    memcpy(TfLiteTensorData(input_tensor), input_mat.ptr<float>(0), num_elements * sizeof(float));

    // Run inference
    auto t0 = std::chrono::high_resolution_clock::now();
    if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk)
        throw std::runtime_error("Hand landmark inference failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Hand: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0 
            << "ms" << std::endl;

    // Output tensors:
    // [0]: landmarks  [1, 63]  - 21 landmarks x (x, y, z)
    // [1]: presence   [1, 1]   - hand presence score (sigmoid)
    // [2]: handedness [1, 1]   - left/right hand score
    const TfLiteTensor* lm_tensor  = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    const TfLiteTensor* pr_tensor  = TfLiteInterpreterGetOutputTensor(interpreter_, 1);

    const float* lm_data = reinterpret_cast<const float*>(TfLiteTensorData(lm_tensor));
    const float* pr_data = reinterpret_cast<const float*>(TfLiteTensorData(pr_tensor));

    float presence = pr_data[0];  // Already a sigmoid score in [0, 1]
    //std::cout << "Hand presence score: " << presence << std::endl;
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
