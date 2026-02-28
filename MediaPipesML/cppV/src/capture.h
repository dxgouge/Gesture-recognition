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

namespace Capture {

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr int    FRAME_WIDTH      = 640;
constexpr int    FRAME_HEIGHT     = 480;
constexpr int    PREDICT_EVERY_N  = 1;
constexpr double TARGET_FPS       = 30.0;
constexpr int    FRAME_DELAY_MS   = static_cast<int>(1000.0 / TARGET_FPS);
constexpr int    NUM_LANDMARKS    = 21;
constexpr int    COORDS_PER_MSG   = NUM_LANDMARKS * 2; // 42 doubles

// Unix socket path — must match landmark_server.py
const std::string SOCKET_PATH = "/tmp/rps_landmarks.sock";
const std::string WINDOW_NAME = "RPS Classifier";

// Gesture colors in BGR
const cv::Scalar COLOR_ROCK     = cv::Scalar(80,  80,  220);
const cv::Scalar COLOR_PAPER    = cv::Scalar(80,  200, 80);
const cv::Scalar COLOR_SCISSORS = cv::Scalar(220, 160, 50);
const cv::Scalar COLOR_UNKNOWN  = cv::Scalar(120, 120, 120);

// =============================================================================
// LANDMARK MESSAGE
// Mirrors what Python sends:
//   1 byte  — hand_detected flag
//   336 bytes — 42 doubles (x0..x20, y0..y20) if detected
// =============================================================================
struct LandmarkMessage {
    bool   hand_detected;
    double x_coords[NUM_LANDMARKS];
    double y_coords[NUM_LANDMARKS];
};

// =============================================================================
// FUNCTIONS
// =============================================================================

cv::Scalar getGestureColor(const std::string& gesture_name);

void drawOverlay(
    cv::Mat& frame,
    const std::string& gesture_name,
    double confidence,
    int buffer_size,
    int window_size
);

// Connect to the Python landmark server
// Returns the socket file descriptor (fd) — a C integer handle to the socket
// Like Python's socket.connect() but returns a raw fd
int connectToServer(const std::string& socket_path);

// Receive one landmark message from the server
// Returns true if a complete message was received
bool receiveLandmarks(int socket_fd, LandmarkMessage& out_msg);

// Run the full real-time loop
void runLoop(Inference::RockPaperScissorsClassifier& classifier);

} // namespace Capture