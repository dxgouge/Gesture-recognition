#include "capture.h"
#include <iostream>
#include <stdexcept>

namespace Capture {

cv::Scalar getGestureColor(const std::string& gesture_name) {
    if      (gesture_name == "Rock")     return COLOR_ROCK;
    else if (gesture_name == "Paper")    return COLOR_PAPER;
    else if (gesture_name == "Scissors") return COLOR_SCISSORS;
    else                                 return COLOR_UNKNOWN;
}

void drawOverlay(
    cv::Mat& frame,
    const std::string& gesture_name,
    double confidence,
    int buffer_size,
    int window_size,
    const Features::LandmarkData& landmarks,
    const std::vector<rps::PalmDetection>& palms)
{
    cv::Scalar color = getGestureColor(gesture_name);

    cv::rectangle(frame, cv::Point(10, 10), cv::Point(320, 90),
                  cv::Scalar(20, 20, 20), -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(320, 90),
                  color, 2);

    cv::putText(frame, gesture_name,
                cv::Point(20, 58),
                cv::FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv::LINE_AA);

    std::string conf_str = std::to_string(static_cast<int>(confidence * 100)) + "%";
    cv::putText(frame, conf_str,
                cv::Point(210, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);

    int bar_w = static_cast<int>(
        (static_cast<double>(buffer_size) / window_size) * 290
    );
    cv::rectangle(frame, cv::Point(10, 75), cv::Point(300, 85),
                  cv::Scalar(40, 40, 40), -1);
    if (bar_w > 0) {
        cv::rectangle(frame, cv::Point(10, 75), cv::Point(10 + bar_w, 85),
                      color, -1);
    }

    std::string buf_str = "Buffer " + std::to_string(buffer_size)
                        + "/" + std::to_string(window_size);

    // Draw Landmarks (for debugging) — only if we have valid landmarks to avoid out-of-bounds access
    if (landmarks.x_coords.size() > 0 && landmarks.y_coords.size() > 0 && !palms.empty()) {
        std::cout << "Drawing landmarks\n: landmarks.x_coords[0]: " << -1 * landmarks.x_coords[0] * FRAME_WIDTH << "\n"
                     "landmarks.y_coords[0]: " << -1 * landmarks.y_coords[0] * FRAME_HEIGHT  << "\n"
                     "FRAME: " << FRAME_WIDTH << "x" << FRAME_HEIGHT << "\n";
        cv::circle(frame, cv::Point( -1 * landmarks.x_coords[0] * FRAME_WIDTH, -1 * landmarks.y_coords[0] * FRAME_HEIGHT), 5, cv::Scalar(0, 255, 255), 2);
    }
    cv::putText(frame, buf_str,
                cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(160, 160, 160), 1);
}

void runLoop(
    Inference::RockPaperScissorsClassifier& classifier,
    rps::PalmDetector&                      palm_detector,
    rps::HandLandmarker&                    hand_landmarker)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        throw std::runtime_error("Could not open webcam");

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    

    std::cout << "Press 'q' or ESC to quit." << std::endl;

    std::deque<std::vector<double>> frame_buffer;
    std::string current_gesture    = "Unknown";
    double      current_confidence = 0.0;
    int         frame_count        = 0;
    cv::Mat     frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);
        frame_count++;

        // Run palm detection
        std::vector<rps::PalmDetection> palms = palm_detector.detect(frame);
        Features::LandmarkData landmarks;
        
        if (!palms.empty()) {
            // Take the highest confidence palm
            const rps::PalmDetection& palm = palms[0];

            // Run landmark detection on the best palm
            rps::LandmarkResult lm_result = hand_landmarker.detect(frame, palm);

            if (lm_result.valid) {
                // Pack into LandmarkData for feature computation
                
                for (int i = 0; i < 21; ++i) {
                    landmarks.x_coords.push_back(
                        static_cast<double>(lm_result.landmarks[i * 2    ]));
                    landmarks.y_coords.push_back(
                        static_cast<double>(lm_result.landmarks[i * 2 + 1]));
                }

                Features::FrameFeatures frame_features;
                bool ok = Features::computeFeatures(landmarks, frame_features);

                if (ok) {
                    std::vector<double> flat = Features::flattenFeatures(frame_features);
                    frame_buffer.push_back(flat);

                    while (static_cast<int>(frame_buffer.size()) > Inference::WINDOW_SIZE)
                        frame_buffer.pop_front();

                    if (static_cast<int>(frame_buffer.size()) == Inference::WINDOW_SIZE &&
                        frame_count % PREDICT_EVERY_N == 0)
                    {
                        std::vector<double> window;
                        window.reserve(Inference::WINDOW_SIZE * flat.size());
                        for (const auto& f : frame_buffer)
                            window.insert(window.end(), f.begin(), f.end());

                        try {
                            Inference::PredictionResult result = classifier.predict(window);
                            current_gesture    = result.gesture_name;
                            current_confidence = result.confidence;
                        } catch (const std::exception& e) {
                            std::cerr << "Prediction error: " << e.what() << std::endl;
                        }
                    }
                }
            }
        } else {
            // No hand detected — clear buffer
            frame_buffer.clear();
            current_gesture    = "Unknown";
            current_confidence = 0.0;
        }

        drawOverlay(frame, current_gesture, current_confidence,
                    static_cast<int>(frame_buffer.size()), Inference::WINDOW_SIZE, landmarks, palms);

        cv::imshow(WINDOW_NAME, frame);

        int key = cv::waitKey(FRAME_DELAY_MS) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Shutting down." << std::endl;
}

} // namespace Capture
