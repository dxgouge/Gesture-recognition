#include "collect.h"
#include "features.h"
#include "capture.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace Collect {

// =============================================================================
// CSV HEADER
// The training script drops:
//   - timestamp, gesture_type          (DROP_COLUMNS)
//   - landmark_*                       (startswith "landmark")
//   - dirV_baseline1_x, _y             (startswith "dirV_baseline")
// Everything else becomes a feature. We write dirV_baseline1 anyway so the
// files are structurally identical — the training script drops it as normal.
// =============================================================================
static void writeHeader(std::ofstream& file) {
    file << "timestamp,"
         << "gesture_type,"
         // Landmark coordinates — 21 × x then 21 × y
         << "landmark_0_x,landmark_1_x,landmark_2_x,landmark_3_x,landmark_4_x,"
         << "landmark_5_x,landmark_6_x,landmark_7_x,landmark_8_x,landmark_9_x,"
         << "landmark_10_x,landmark_11_x,landmark_12_x,landmark_13_x,landmark_14_x,"
         << "landmark_15_x,landmark_16_x,landmark_17_x,landmark_18_x,landmark_19_x,"
         << "landmark_20_x,"
         << "landmark_0_y,landmark_1_y,landmark_2_y,landmark_3_y,landmark_4_y,"
         << "landmark_5_y,landmark_6_y,landmark_7_y,landmark_8_y,landmark_9_y,"
         << "landmark_10_y,landmark_11_y,landmark_12_y,landmark_13_y,landmark_14_y,"
         << "landmark_15_y,landmark_16_y,landmark_17_y,landmark_18_y,landmark_19_y,"
         << "landmark_20_y,"
         // Angles
         << "angle_index,"
         << "angle_middle,"
         << "angle_ring,"
         << "angle_pinky,"
         << "angle_baseline1,"
         << "angle_middle_to_ring,"
         << "angle_base_middle_to_ring,"
         << "angle_middle_to_baseline1,"
         // Distances
         << "dis_index_tip_to_base,"
         << "dis_middle_tip_to_base,"
         << "dis_ring_tip_to_base,"
         << "dis_pinky_tip_to_base,"
         << "distancesAggregated,"
         // Direction vectors
         // dirV_baseline1 IS written (training script drops it — but must be
         << "dirV_index_x,"      << "dirV_index_y,"
         << "dirV_middle_x,"     << "dirV_middle_y,"
         << "dirV_ring_x,"       << "dirV_ring_y,"
         << "dirV_pinky_x,"      << "dirV_pinky_y,"
         << "dirV_baseline1_x,"  << "dirV_baseline1_y,"
         << "dirV_middle_base_x,"<< "dirV_middle_base_y,"
         << "dirV_ring_base_x,"  << "dirV_ring_base_y"
         << "\n";
}

static void writeRow(
    std::ofstream&                  file,
    int                             timestamp_s,
    const std::string&              gesture_name,
    const Features::LandmarkData&   landmarks,
    const Features::FrameFeatures&  f)
{
    file << std::fixed << std::setprecision(8);

    file << timestamp_s   << ","
         << gesture_name  << ",";

    // Landmark x coords
    for (int i = 0; i < 21; ++i)
        file << landmarks.x_coords[i] << ",";

    // Landmark y coords — last one has no trailing comma before the angles
    for (int i = 0; i < 21; ++i)
        file << landmarks.y_coords[i] << ",";

    // Angles
    file << f.angles.angle_index               << ","
         << f.angles.angle_middle              << ","
         << f.angles.angle_ring                << ","
         << f.angles.angle_pinky               << ","
         << f.angles.angle_baseline1           << ","
         << f.angles.angle_middle_to_ring      << ","
         << f.angles.angle_base_middle_to_ring << ","
         << f.angles.angle_middle_to_baseline1 << ",";

    // Distances
    file << f.distances.dis_index_tip_to_base  << ","
         << f.distances.dis_middle_tip_to_base << ","
         << f.distances.dis_ring_tip_to_base   << ","
         << f.distances.dis_pinky_tip_to_base  << ","
         << f.distances_aggregated             << ",";

    // Vectors 
    file << f.vectors.dirV_index[0]       << "," << f.vectors.dirV_index[1]       << ","
         << f.vectors.dirV_middle[0]      << "," << f.vectors.dirV_middle[1]      << ","
         << f.vectors.dirV_ring[0]        << "," << f.vectors.dirV_ring[1]        << ","
         << f.vectors.dirV_pinky[0]       << "," << f.vectors.dirV_pinky[1]       << ","
         << f.vectors.dirV_baseline1[0]   << "," << f.vectors.dirV_baseline1[1]   << ","
         << f.vectors.dirV_middle_base[0] << "," << f.vectors.dirV_middle_base[1] << ","
         << f.vectors.dirV_ring_base[0]   << "," << f.vectors.dirV_ring_base[1]
         << "\n";
}

// =============================================================================
// HUD OVERLAY
// =============================================================================
static void drawCollectOverlay(
    cv::Mat&           frame,
    const std::string& gesture_name,
    bool               recording,
    int                rows_written,
    bool               hand_detected)
{
    cv::Scalar border_color = recording
        ? cv::Scalar(0, 0, 220)
        : cv::Scalar(80, 80, 80);

    cv::rectangle(frame, cv::Point(10, 10), cv::Point(440, 105),
                  cv::Scalar(20, 20, 20), -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(440, 105),
                  border_color, 2);

    cv::putText(frame,
                "Gesture: " + gesture_name,
                cv::Point(20, 48),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(220, 220, 220), 2, cv::LINE_AA);

    std::string status;
    cv::Scalar  status_color;

    if (recording) {
        status       = "REC  [SPACE stop]  rows: " + std::to_string(rows_written);
        status_color = cv::Scalar(80, 80, 220);
    } else if (hand_detected) {
        status       = "Ready  [SPACE to record]";
        status_color = cv::Scalar(80, 200, 80);
    } else {
        status       = "No hand detected";
        status_color = cv::Scalar(120, 120, 120);
    }

    cv::putText(frame, status,
                cv::Point(20, 85),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                status_color, 1, cv::LINE_AA);
}

// =============================================================================
// MAIN COLLECTION LOOP
// Single-threaded by design — every frame is processed in order and written
// immediately. Drops would create gaps in training data, so we don't want
// the async / drop-oldest architecture used in the classifier.
// =============================================================================
void runCollect(
    const std::string&   gesture_name,
    const std::string&   output_path,
    rps::PalmDetector&   palm_detector,
    rps::HandLandmarker& hand_landmarker)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        throw std::runtime_error("Could not open webcam");

    cap.set(cv::CAP_PROP_FOURCC,       cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  Capture::FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Capture::FRAME_HEIGHT);
    cap.set(cv::CAP_PROP_FPS,          30);

    // Append to existing file, write header only if file is new
    bool file_exists = std::ifstream(output_path).good();
    std::ofstream csv(output_path, std::ios::app);
    if (!csv.is_open())
        throw std::runtime_error("Could not open output file: " + output_path);
    if (!file_exists)
        writeHeader(csv);

    auto start_time = std::chrono::steady_clock::now();
    bool recording    = false;
    int  rows_written = 0;
    cv::Mat frame;

    std::cout << "\n=== Collection Mode ===\n"
              << "Gesture : " << gesture_name << "\n"
              << "Output  : " << output_path  << "\n"
              << "SPACE   : start / stop recording\n"
              << "Q / ESC : quit and save\n\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);

        std::vector<rps::PalmDetection> palms = palm_detector.detect(frame);
        bool hand_detected = false;

        if (!palms.empty()) {
            rps::LandmarkResult lm = hand_landmarker.detect(frame, palms[0]);

            if (lm.valid) {
                hand_detected = true;

                // Build LandmarkData
                Features::LandmarkData landmarks;
                for (int i = 0; i < 21; ++i) {
                    landmarks.x_coords.push_back(
                        static_cast<double>(lm.landmarks[i * 2    ]));
                    landmarks.y_coords.push_back(
                        static_cast<double>(lm.landmarks[i * 2 + 1]));
                }

                if (recording) {
                    Features::FrameFeatures feat;
                    if (Features::computeFeatures(landmarks, feat)) {
                        auto now = std::chrono::steady_clock::now();
                        int ts   = static_cast<int>(
                            std::chrono::duration_cast<std::chrono::seconds>(
                                now - start_time).count());

                        writeRow(csv, ts, gesture_name, landmarks, feat);
                        csv.flush();  // flush every row — safe if process is killed early
                        ++rows_written;
                    }
                }

                // Draw landmarks
                for (int i = 0; i < 21; ++i) {
                    int x = static_cast<int>(landmarks.x_coords[i] * Capture::FRAME_WIDTH);
                    int y = static_cast<int>(landmarks.y_coords[i] * Capture::FRAME_HEIGHT);
                    cv::circle(frame, cv::Point(x, y), 4, cv::Scalar(0, 255, 255), 2);
                }
            }
        }

        drawCollectOverlay(frame, gesture_name, recording, rows_written, hand_detected);
        cv::imshow("RPS Data Collection", frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == ' ') {
            recording = !recording;
            std::cout << (recording ? "Recording started...\n"
                                    : "Paused. Rows so far: " + std::to_string(rows_written) + "\n");
        }
    }

    csv.close();
    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nDone. Wrote " << rows_written
              << " rows to " << output_path << "\n";
}

} // namespace Collect