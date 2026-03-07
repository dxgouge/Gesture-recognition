#pragma once

#include <string>
#include <deque>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "features.h"
#include "inference.h"
#include "palm_detector.h"
#include "hand_landmarker.h"

namespace Capture {

constexpr int    FRAME_WIDTH            = 640;
constexpr int    FRAME_HEIGHT           = 480;
constexpr double TARGET_FPS             = 40.0;
constexpr int    FRAME_DELAY_MS         = 1;
constexpr int    FRAME_QUEUE_CAPACITY   = 2;   // drop stale frames, not accumulate them
constexpr int    RESULT_QUEUE_CAPACITY  = 4;

const std::string WINDOW_NAME = "RPS Classifier";

const cv::Scalar COLOR_ROCK     = cv::Scalar(80,  80,  220);
const cv::Scalar COLOR_PAPER    = cv::Scalar(80,  200, 80);
const cv::Scalar COLOR_SCISSORS = cv::Scalar(220, 160, 50);
const cv::Scalar COLOR_UNKNOWN  = cv::Scalar(120, 120, 120);

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe bounded queue
//
// When full, drops the OLDEST item (not the newest) so the inference thread
// always works on the most recent frame. This trades a small number of dropped
// frames for bounded, predictable latency — no unbounded backlog ever builds.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(int capacity) : capacity_(capacity) {}

    // Push item. Evicts oldest entry if at capacity.
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (static_cast<int>(queue_.size()) >= capacity_) {
            queue_.pop();
            ++dropped_;
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    // Block until an item is available or stop() is called.
    // Returns nullopt as a sentinel telling the consumer thread to exit.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Non-blocking pop. Returns nullopt immediately if queue is empty.
    std::optional<T> try_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Wake all blocked consumers so they can exit cleanly.
    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

    int droppedCount() const { return dropped_.load(); }

private:
    std::queue<T>           queue_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    int                     capacity_;
    std::atomic<int>        dropped_{0};
    bool                    stopped_{false};
};

// ─────────────────────────────────────────────────────────────────────────────
// Inter-thread message types
// ─────────────────────────────────────────────────────────────────────────────

// Capture thread  →  Inference thread
struct FramePacket {
    cv::Mat frame;        // deep copy — capture thread can immediately grab next frame
    int     frame_index;
};

// Inference thread  →  Display thread
struct ResultPacket {
    cv::Mat     frame;
    std::string gesture_name;
    double      confidence;
    int         buffer_fill;
    Features::LandmarkData          landmarks;
    std::vector<rps::PalmDetection> palms;
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

cv::Scalar getGestureColor(const std::string& gesture_name);

void drawOverlay(
    cv::Mat& frame,
    const std::string& gesture_name,
    double confidence,
    int buffer_size,
    int window_size,
    const Features::LandmarkData& landmarks,
    const std::vector<rps::PalmDetection>& palms
);

void runLoop(
    Inference::RockPaperScissorsClassifier& classifier,
    rps::PalmDetector&                      palm_detector,
    rps::HandLandmarker&                    hand_landmarker
);

} // namespace Capture