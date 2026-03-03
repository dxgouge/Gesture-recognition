#include <iostream>
#include <stdexcept>
#include "inference.h"
#include "capture.h"
#include "palm_detector.h"
#include "hand_landmarker.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./rps_cpp <lgbm_model.txt> <palm_detection.tflite> <hand_landmark.tflite>" << std::endl;
        std::cerr << "Example: ./rps_cpp ../rps_lgbm_model.txt ../palm_detection_lite.tflite ../hand_landmark_lite.tflite" << std::endl;
        return 1;
    }

    try {
        Inference::RockPaperScissorsClassifier classifier(argv[1]);
        rps::PalmDetector    palm_detector(argv[2]);
        rps::HandLandmarker  hand_landmarker(argv[3]);

        Capture::runLoop(classifier, palm_detector, hand_landmarker);

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
