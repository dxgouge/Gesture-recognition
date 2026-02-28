#pragma once
// ^^ This is a "header guard" — it tells the compiler to only include this file
//    once even if multiple files reference it. Python has no equivalent since
//    imports are handled automatically.

#include <vector>   // std::vector — like Python list
#include <array>    // std::array  — fixed size list
#include <string>   // std::string — like Python str

// =============================================================================
// NAMESPACE
// In Python you'd put these in a class or module. In C++ we use a namespace
// to group related functions together without making a full class.
// Usage: Features::getLandmarkDistance(...)
// =============================================================================
namespace Features {

// =============================================================================
// DATA STRUCTURES
// Like Python dataclasses — groups related data together.
// 'struct' members are public by default (unlike 'class').
// =============================================================================

// Holds the raw x/y coordinates of all 21 MediaPipe landmarks
struct LandmarkData {
    std::vector<double> x_coords;  // 21 values
    std::vector<double> y_coords;  // 21 values
};

// Holds a 2D direction vector (x, y)
// std::array<double, 2> means: a fixed array of exactly 2 doubles
using Vec2 = std::array<double, 2>;

// Holds all 8 computed angles
struct Angles {
    double angle_index;
    double angle_middle;
    double angle_ring;
    double angle_pinky;
    double angle_baseline1;
    double angle_middle_to_ring;
    double angle_base_middle_to_ring;
    double angle_middle_to_baseline1;
};

// Holds all 4 finger tip-to-base distances
struct Distances {
    double dis_index_tip_to_base;
    double dis_middle_tip_to_base;
    double dis_ring_tip_to_base;
    double dis_pinky_tip_to_base;
};

// Holds all 7 direction vectors
struct Vectors {
    Vec2 dirV_index;
    Vec2 dirV_middle;
    Vec2 dirV_ring;
    Vec2 dirV_pinky;
    Vec2 dirV_baseline1;
    Vec2 dirV_middle_base;
    Vec2 dirV_ring_base;
};

// The full feature set for one frame — everything the model needs
// This is the C++ equivalent of the data dict returned by get_latest_data()
struct FrameFeatures {
    double      palm_size;
    double      scale_factor;
    Angles      angles;
    Distances   distances;
    double      distances_aggregated;
    Vectors     vectors;
};

// =============================================================================
// FUNCTION DECLARATIONS
// In Python you just define functions. In C++ the header declares them
// (says they exist and what types they take/return), and the .cpp defines them
// (says what they actually do). The compiler needs to see the declaration
// before it can compile any code that calls the function.
// =============================================================================

// Calculate Euclidean distance between two landmarks
// 'const' means the function promises not to modify these parameters
// '&' means pass by reference (no copy) — more efficient than Python's default
double getLandmarkDistance(
    int landmark1,
    int landmark2,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords
);

// Calculate direction vector from landmark1 to landmark2
Vec2 getFingerVector(
    int landmark1,
    int landmark2,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords
);

// Calculate angle of a vector relative to a reference vector (in degrees)
// This is the rotation-invariant version we discussed
double getFingerAngle(
    const Vec2& vector,
    const Vec2& reference_vector
);

// Calculate angle between two vectors (in degrees)
double getAngleBetweenFingers(
    const Vec2& vector1,
    const Vec2& vector2
);

// Calculate the aggregated mean pairwise distance across all landmarks,
// scaled to palm size
double getDistancesAggregated(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    double scale
);

// Master function — takes raw landmarks, returns a fully computed FrameFeatures
// Returns false if palm_size is 0 (bad frame), true on success
// 'out_features' is an output parameter — the function fills it in
// In Python you'd just return a dict; in C++ we pass in a reference to fill
bool computeFeatures(
    const LandmarkData& landmarks,
    FrameFeatures& out_features
);

// Flatten a FrameFeatures into a vector in the exact order the model was trained on:
//   angle_index, angle_middle, angle_ring, angle_pinky, angle_baseline1,
//   angle_middle_to_ring, angle_base_middle_to_ring, angle_middle_to_baseline1,
//   dis_index_tip_to_base, dis_middle_tip_to_base, dis_ring_tip_to_base, dis_pinky_tip_to_base,
//   distancesAggregated,
//   dirV_index_x, dirV_index_y, dirV_middle_x, dirV_middle_y,
//   dirV_ring_x, dirV_ring_y, dirV_pinky_x, dirV_pinky_y,
//   dirV_middle_base_x, dirV_middle_base_y, dirV_ring_base_x, dirV_ring_base_y
// = 25 features per frame, 125 total for a 5-frame window
// NOTE: dirV_baseline1 is excluded (dropped during training), angle_baseline1 is included
std::vector<double> flattenFeatures(const FrameFeatures& features);

} // namespace Features