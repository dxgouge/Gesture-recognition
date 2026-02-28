#include <iostream>
#include <stdexcept>

#include "inference.h"
#include "capture.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./rps_cpp <path_to_model.txt>" << std::endl;
        std::cerr << "Example: ./rps_cpp ../rps_lgbm_model.txt" << std::endl;
        return 1;
    }

    try {
        // Load model
        Inference::RockPaperScissorsClassifier classifier(argv[1]);

        // Run the real-time loop
        // This blocks until the user presses 'q' or ESC
        Capture::runLoop(classifier);

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}