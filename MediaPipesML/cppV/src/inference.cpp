#include "inference.h"

#include <stdexcept>  // std::runtime_error
#include <algorithm>  // std::max_element
#include <iostream>   // std::cout, std::cerr
#include <fstream>    // std::ifstream — for reading the model file
#include <sstream>    // std::istreambuf_iterator

namespace Inference {

// =============================================================================
// CONSTRUCTOR
// =============================================================================

RockPaperScissorsClassifier::RockPaperScissorsClassifier(const std::string& model_path)
    : model_(nullptr)
    , num_features_(0)
{
    // This version of the LightGBM C API does not have LGBM_BoosterLoadModelFromFile
    // so we read the file into a string ourselves and use LoadModelFromString instead
    std::ifstream model_file(model_path);
    if (!model_file.is_open()) {
        throw std::runtime_error("Could not open model file: " + model_path);
    }

    // Read entire file into a string
    // istreambuf_iterator reads the file character by character into the string
    std::string model_str(
        (std::istreambuf_iterator<char>(model_file)),
         std::istreambuf_iterator<char>()
    );
    model_file.close();

    int num_iterations = 0;
    int result = LGBM_BoosterLoadModelFromString(
        model_str.c_str(),  // C-style string pointer to model contents
        &num_iterations,    // output: number of iterations loaded
        &model_             // output: filled with our model handle
    );

    if (result != 0) {
        throw std::runtime_error(
            std::string("Failed to load LightGBM model from: ") + model_path
        );
    }

    // LGBM_BoosterGetNumFeature expects int* not int64_t*
    // The error told us exactly this — always read compiler errors carefully,
    // they tell you the exact type mismatch
    int num_features_out = 0;
    LGBM_BoosterGetNumFeature(model_, &num_features_out);
    num_features_ = num_features_out;

    std::cout << "Model loaded: " << model_path         << std::endl;
    std::cout << "  Iterations: " << num_iterations     << std::endl;
    std::cout << "  Features:   " << num_features_      << std::endl;
}

// =============================================================================
// DESTRUCTOR
// =============================================================================

RockPaperScissorsClassifier::~RockPaperScissorsClassifier() {
    if (model_ != nullptr) {
        LGBM_BoosterFree(model_);
        model_ = nullptr;
    }
}

// =============================================================================
// PREDICT
// =============================================================================

PredictionResult RockPaperScissorsClassifier::predict(
    const std::vector<double>& feature_window) const
{
    if (static_cast<int>(feature_window.size()) != num_features_) {
        throw std::runtime_error(
            "Feature window size mismatch. Expected: " +
            std::to_string(num_features_) +
            " Got: " +
            std::to_string(feature_window.size())
        );
    }

    std::vector<double> probabilities(NUM_CLASSES, 0.0);
    int64_t out_len = 0;

    int result = LGBM_BoosterPredictForMat(
        model_,
        feature_window.data(),
        C_API_DTYPE_FLOAT64,
        1,                    // 1 row
        num_features_,        // n columns
        1,                    // row-major
        C_API_PREDICT_NORMAL,
        0,                    // start iteration
        -1,                   // end iteration (-1 = use all)
        "",                   // extra params
        &out_len,
        probabilities.data()
    );

    if (result != 0) {
        throw std::runtime_error("LightGBM prediction failed");
    }

    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    int predicted_class = static_cast<int>(
        std::distance(probabilities.begin(), max_it)
    );

    PredictionResult prediction;
    prediction.predicted_class = predicted_class;
    prediction.confidence      = probabilities[predicted_class];
    prediction.gesture_name    = GESTURE_NAMES[predicted_class];

    std::copy(
        probabilities.begin(),
        probabilities.end(),
        prediction.probabilities.begin()
    );

    return prediction;
}

// =============================================================================
// GETTERS
// =============================================================================

int RockPaperScissorsClassifier::getNumFeatures() const {
    return num_features_;
}

} // namespace Inference