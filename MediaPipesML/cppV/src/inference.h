#pragma once

#include <string>
#include <vector>
#include <array>


#include <LightGBM/c_api.h>

namespace Inference {

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr int NUM_CLASSES  = 3;
constexpr int WINDOW_SIZE  = 5;

// Gesture names indexed by class label
// std::array<const char*, 3> is a fixed array of 3 string literals
constexpr std::array<const char*, NUM_CLASSES> GESTURE_NAMES = {
    "Rock", "Paper", "Scissors"
};

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Result of one prediction
struct PredictionResult {
    int    predicted_class;                       // 0, 1, or 2
    double confidence;                            // probability of predicted class
    std::array<double, NUM_CLASSES> probabilities; // all 3 class probabilities
    const char* gesture_name;                     // "Rock", "Paper", "Scissors"
};

// =============================================================================
// CLASSIFIER CLASS
// =============================================================================

class RockPaperScissorsClassifier {
public:
    // Constructor — loads the model from disk
    // 'explicit' prevents accidental implicit conversions
    explicit RockPaperScissorsClassifier(const std::string& model_path);

    // Destructor — frees the model from memory
    // '~' prefix marks this as a destructor
    ~RockPaperScissorsClassifier();

    // Disable copying — the LightGBM handle can't be safely copied
    // This is a C++11 pattern to explicitly prevent copy construction
    RockPaperScissorsClassifier(const RockPaperScissorsClassifier&)            = delete;
    RockPaperScissorsClassifier& operator=(const RockPaperScissorsClassifier&) = delete;

    // Predict on a flattened feature window
    // Input:  flattened vector of WINDOW_SIZE frames × n_features doubles
    // Output: PredictionResult with class, confidence, and all probabilities
    PredictionResult predict(const std::vector<double>& feature_window) const;

    // Returns the number of features the model expects per window
    int getNumFeatures() const;

private:
    // BoosterHandle is LightGBM's opaque pointer type for the loaded model
    // 'void*' under the hood — a raw C pointer
    // This is what C APIs look like — you hold a handle and pass it to functions
    BoosterHandle model_;

    int num_features_;  // cached from the model at load time
};

} // namespace Inference