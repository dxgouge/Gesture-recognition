#pragma once

#include <string>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "features.h"
#include "inference.h"
#include "palm_detector.h"
#include "hand_landmarker.h"

namespace Capture {

constexpr int    FRAME_WIDTH     = 640;
constexpr int    FRAME_HEIGHT    = 480;
constexpr int    PREDICT_EVERY_N = 1;
constexpr double TARGET_FPS      = 30.0;
constexpr int    FRAME_DELAY_MS  = static_cast<int>(1000.0 / TARGET_FPS);

const std::string WINDOW_NAME = "RPS Classifier";

const cv::Scalar COLOR_ROCK     = cv::Scalar(80,  80,  220);
const cv::Scalar COLOR_PAPER    = cv::Scalar(80,  200, 80);
const cv::Scalar COLOR_SCISSORS = cv::Scalar(220, 160, 50);
const cv::Scalar COLOR_UNKNOWN  = cv::Scalar(120, 120, 120);

cv::Scalar getGestureColor(const std::string& gesture_name);

void drawOverlay(
    cv::Mat& frame,
    const std::string& gesture_name,
    double confidence,
    int buffer_size,
    int window_size
);

void runLoop(
    Inference::RockPaperScissorsClassifier& classifier,
    rps::PalmDetector&                      palm_detector,
    rps::HandLandmarker&                    hand_landmarker
);

} // namespace Capture
