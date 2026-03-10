#pragma once
#include <string>
#include "palm_detector.h"
#include "hand_landmarker.h"

namespace Collect {

// Runs data collection mode.
// gesture_name: "rock", "paper", or "scissors" — written into gesture_type column
// output_path:  path to output CSV (appends if exists, creates with header if not)
void runCollect(
    const std::string&   gesture_name,
    const std::string&   output_path,
    rps::PalmDetector&   palm_detector,
    rps::HandLandmarker& hand_landmarker
);

} // namespace Collect
