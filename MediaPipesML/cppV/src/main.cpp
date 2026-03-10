#include <iostream>
#include <stdexcept>
#include <string>
#include "inference.h"
#include "capture.h"
#include "collect.h"
#include "palm_detector.h"
#include "hand_landmarker.h"

static void printUsage(const char* prog) {
    std::cerr << "\nUsage:\n"
              << "  Classify:  " << prog
              << " classify <lgbm.txt> <palm.tflite> <landmark.tflite>\n"
              << "  Collect:   " << prog
              << " collect <rock|paper|scissors> <output.csv> <palm.tflite> <landmark.tflite>\n\n";

              //./rps_cpp classify ../../rps_lgbm_model.txt ~/palm_detection_lite.tflite ~/hand_landmark_lite.tflite
              //./rps_cpp collect rock ../data/rock_data.csv ~/palm_detection_lite.tflite ~/hand_landmark_lite.tflite
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    const std::string mode = argv[1];

    try {
        if (mode == "classify") {
            if (argc < 5) { printUsage(argv[0]); return 1; }

            Inference::RockPaperScissorsClassifier classifier(argv[2]);
            rps::PalmDetector    palm_detector(argv[3]);
            rps::HandLandmarker  hand_landmarker(argv[4]);
            Capture::runLoop(classifier, palm_detector, hand_landmarker);

        } else if (mode == "collect") {
            if (argc < 6) { printUsage(argv[0]); return 1; }

            const std::string gesture = argv[2];
            const std::string outpath = argv[3];

            if (gesture != "rock" && gesture != "paper" && gesture != "scissors") {
                std::cerr << "Error: gesture must be rock, paper, or scissors\n";
                return 1;
            }

            rps::PalmDetector    palm_detector(argv[4]);
            rps::HandLandmarker  hand_landmarker(argv[5]);
            Collect::runCollect(gesture, outpath, palm_detector, hand_landmarker);

        } else {
            std::cerr << "Error: unknown mode '" << mode << "'\n";
            printUsage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}