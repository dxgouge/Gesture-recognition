#include "capture.h"
#include <iostream>
#include <stdexcept>
#include <chrono>

namespace Capture {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────

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
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(320, 90), color, 2);

    cv::putText(frame, gesture_name,
                cv::Point(20, 58),
                cv::FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv::LINE_AA);

    std::string conf_str = std::to_string(static_cast<int>(confidence * 100)) + "%";
    cv::putText(frame, conf_str,
                cv::Point(210, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);

    int bar_w = static_cast<int>(
        (static_cast<double>(buffer_size) / window_size) * 290);
    cv::rectangle(frame, cv::Point(10, 75), cv::Point(300, 85),
                  cv::Scalar(40, 40, 40), -1);
    if (bar_w > 0)
        cv::rectangle(frame, cv::Point(10, 75), cv::Point(10 + bar_w, 85), color, -1);

    std::string buf_str = "Buffer " + std::to_string(buffer_size)
                        + "/" + std::to_string(window_size);
    cv::putText(frame, buf_str,
                cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(160, 160, 160), 1);

    // Landmarks
    if (!landmarks.x_coords.empty() && !palms.empty()) {
        for (size_t i = 0; i < landmarks.x_coords.size(); ++i) {
            int x = static_cast<int>(landmarks.x_coords[i] * FRAME_WIDTH);
            int y = static_cast<int>(landmarks.y_coords[i] * FRAME_HEIGHT);
            cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 255), 2);
        }
    }

    // Palm bounding boxes
    for (const auto& palm : palms) {
        int cx = static_cast<int>(palm.cx * FRAME_WIDTH);
        int cy = static_cast<int>(palm.cy * FRAME_HEIGHT);
        int w  = static_cast<int>(palm.width  * FRAME_WIDTH);
        int h  = static_cast<int>(palm.height * FRAME_HEIGHT);
        cv::rectangle(frame,
            cv::Point(cx - w/2, cy - h/2),
            cv::Point(cx + w/2, cy + h/2),
            cv::Scalar(0, 165, 255), 2);

        float cos_r = std::cos(palm.rotation);
        float sin_r = std::sin(palm.rotation);
        float half  = std::max(palm.width * FRAME_WIDTH,
                               palm.height * FRAME_HEIGHT) * 2.6f * 0.5f;
        float shift_y = -0.5f * half;
        float rcx = cx + (-sin_r * shift_y);
        float rcy = cy + ( cos_r * shift_y);

        cv::Point2f corners[4];
        float dirs[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
        for (int i = 0; i < 4; ++i) {
            float lx = dirs[i][0] * half;
            float ly = dirs[i][1] * half;
            corners[i] = cv::Point2f(
                rcx + cos_r * lx - sin_r * ly,
                rcy + sin_r * lx + cos_r * ly);
        }
        for (int i = 0; i < 4; ++i)
            cv::line(frame, corners[i], corners[(i+1)%4], cv::Scalar(0, 255, 0), 2);
        cv::circle(frame,
            cv::Point(static_cast<int>(rcx), static_cast<int>(rcy)),
            5, cv::Scalar(0, 255, 0), -1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference thread body
//
// Owns the sliding window buffer (frame_buffer) and all three models.
// Pops FramePackets, runs palm→landmark→feature→classify, pushes ResultPackets.
//
// Why keep frame_buffer here rather than on the capture thread?
// Because the window must contain temporally adjacent frames. If we built
// the window on the capture side we'd need to ship it across the queue,
// which copies up to WINDOW_SIZE * 125 doubles every frame. Keeping state
// local to inference is cheaper and simpler.
// ─────────────────────────────────────────────────────────────────────────────
static void inferenceThread(
    BoundedQueue<FramePacket>&  in_queue,
    BoundedQueue<ResultPacket>& out_queue,
    Inference::RockPaperScissorsClassifier& classifier,
    rps::PalmDetector&    palm_detector,
    rps::HandLandmarker&  hand_landmarker)
{
    std::deque<std::vector<double>> frame_buffer;
    std::string last_gesture    = "Unknown";
    double      last_confidence = 0.0;
    int         frame_count     = 0;

    while (true) {
        auto packet_opt = in_queue.pop();
        if (!packet_opt) break;   // queue stopped — exit cleanly

        cv::Mat& frame = packet_opt->frame;
        frame_count++;

        std::vector<rps::PalmDetection> palms = palm_detector.detect(frame);
        Features::LandmarkData landmarks;

        if (!palms.empty()) {
            rps::LandmarkResult lm_result = hand_landmarker.detect(frame, palms[0]);

            if (lm_result.valid) {
                for (int i = 0; i < 21; ++i) {
                    landmarks.x_coords.push_back(
                        static_cast<double>(lm_result.landmarks[i * 2    ]));
                    landmarks.y_coords.push_back(
                        static_cast<double>(lm_result.landmarks[i * 2 + 1]));
                }

                Features::FrameFeatures frame_features;
                if (Features::computeFeatures(landmarks, frame_features)) {
                    auto flat = Features::flattenFeatures(frame_features);
                    frame_buffer.push_back(flat);
                    while (static_cast<int>(frame_buffer.size()) > Inference::WINDOW_SIZE)
                        frame_buffer.pop_front();

                    if (static_cast<int>(frame_buffer.size()) == Inference::WINDOW_SIZE
                        && frame_count % 2 == 0)
                    {
                        std::vector<double> window;
                        window.reserve(Inference::WINDOW_SIZE * flat.size());
                        for (const auto& f : frame_buffer)
                            window.insert(window.end(), f.begin(), f.end());

                        try {
                            auto result       = classifier.predict(window);
                            last_gesture      = result.gesture_name;
                            last_confidence   = result.confidence;
                        } catch (const std::exception& e) {
                            std::cerr << "Prediction error: " << e.what() << "\n";
                        }
                    }
                }
            }
        } else {
            frame_buffer.clear();
            last_gesture    = "Unknown";
            last_confidence = 0.0;
        }

        ResultPacket result;
        result.frame        = std::move(frame);   // zero-copy — move the Mat header
        result.gesture_name = last_gesture;
        result.confidence   = last_confidence;
        result.buffer_fill  = static_cast<int>(frame_buffer.size());
        result.landmarks    = landmarks;
        result.palms        = palms;

        out_queue.push(std::move(result));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// runLoop — now three stages running in parallel
//
// Thread layout:
//   Main thread  : capture + display (OpenCV GUI must stay on the main thread
//                  on most platforms — imshow/waitKey are not thread-safe)
//   Worker thread: palm detection + landmark + feature extraction + LGBM
//
// The two BoundedQueues act as the connective tissue.
// ─────────────────────────────────────────────────────────────────────────────
void runLoop(
    Inference::RockPaperScissorsClassifier& classifier,
    rps::PalmDetector&                      palm_detector,
    rps::HandLandmarker&                    hand_landmarker)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        throw std::runtime_error("Could not open webcam");
    cap.set(cv::CAP_PROP_FPS, 120);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    // Shared queues
    BoundedQueue<FramePacket>  frame_queue(FRAME_QUEUE_CAPACITY);
    BoundedQueue<ResultPacket> result_queue(RESULT_QUEUE_CAPACITY);

    // Spin up the inference worker
    std::thread worker(inferenceThread,
        std::ref(frame_queue),
        std::ref(result_queue),
        std::ref(classifier),
        std::ref(palm_detector),
        std::ref(hand_landmarker));

    // ── Display state (updated whenever a ResultPacket arrives) ──────────────
    // The display thread (main) should never block waiting for inference.
    // If no new result is ready, we keep showing the last known overlay.
    std::string current_gesture    = "Unknown";
    double      current_confidence = 0.0;
    int         current_buffer     = 0;
    Features::LandmarkData          current_landmarks;
    std::vector<rps::PalmDetection> current_palms;

    // Non-blocking result drain helper — empties result_queue without waiting.
    // We call this once per capture iteration so the display stays fresh.
    auto drainResults = [&]() {
        // Try to consume all pending results; keep only the latest.
        // Using a raw try-pop pattern so the capture loop is never blocked
        // by a slow inference thread.
        ResultPacket latest;
        bool got_one = false;
        while (true) {
            // Peek: temporarily pop, keep if valid
            // We implement non-blocking pop via a short timed approach:
            // use the queue's internal state — but our BoundedQueue only
            // has a blocking pop(). So we use a separate atomic flag trick:
            // instead, just give result_queue a try_pop via timeout = 0.
            // Simplest correct approach: drain by checking size via a
            // dedicated non-blocking overload.
            //
            // We add try_pop() below — see NOTE in BoundedQueue above.
            auto opt = result_queue.try_pop();
            if (!opt) break;
            latest  = std::move(*opt);
            got_one = true;
        }
        if (got_one) {
            current_gesture    = latest.gesture_name;
            current_confidence = latest.confidence;
            current_buffer     = latest.buffer_fill;
            current_landmarks  = std::move(latest.landmarks);
            current_palms      = std::move(latest.palms);
        }
    };

    std::cout << "Press 'q' or ESC to quit.\n";

    int frame_index = 0;
    auto fps_start  = std::chrono::high_resolution_clock::now();
    int  fps_count  = 0;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);

        // Ship a copy to inference — clone() makes a deep copy so we can
        // immediately flip and display the original without a data race.
        FramePacket pkt;
        pkt.frame       = frame.clone();
        pkt.frame_index = frame_index++;
        frame_queue.push(std::move(pkt));

        // Pull any finished results (non-blocking)
        drainResults();

        // Draw overlay on the live frame using the last known result
        drawOverlay(frame, current_gesture, current_confidence,
                    current_buffer, Inference::WINDOW_SIZE,
                    current_landmarks, current_palms);

        cv::imshow(WINDOW_NAME, frame);

        // FPS display
        ++fps_count;
        if (fps_count % 30 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - fps_start).count() / 1000.0;
            std::cout << "Display FPS: " << fps_count / elapsed
                      << "  |  dropped frames: " << frame_queue.droppedCount()
                      << "\n";
            fps_count = 0;
            fps_start = now;
        }

        int key = cv::waitKey(FRAME_DELAY_MS) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    // Orderly shutdown: signal the worker to stop, then wait for it
    frame_queue.stop();
    worker.join();

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Shutting down.\n";
}

} // namespace Capture