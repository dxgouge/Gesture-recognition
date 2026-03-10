#include "features.h"

#include <cmath>      
#include <numeric>   
#include <stdexcept>  
#include <algorithm>  

// =============================================================================
// CONSTANTS
// =============================================================================
namespace {
    constexpr double REFERENCE_PALM_SIZE = 0.14;
    constexpr double PI = 3.14159265358979323846;

    // Convert radians to degrees — cmath works in radians by default
    // 'inline' suggests the compiler paste this code directly at the call site
    // rather than making a function call, which is faster for tiny functions
    inline double toDegrees(double radians) {
        return radians * (180.0 / PI);
    }
}

// =============================================================================
// IMPLEMENTATION
// Notice we prefix everything with Features:: to tell the compiler these
// functions belong to the Features namespace declared in the header.
// =============================================================================
namespace Features {

// -----------------------------------------------------------------------------
double getLandmarkDistance(
    int landmark1,
    int landmark2,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords)
{
    double dx = x_coords[landmark1] - x_coords[landmark2];
    double dy = y_coords[landmark1] - y_coords[landmark2];
    return std::sqrt(dx * dx + dy * dy);
}

// -----------------------------------------------------------------------------
Vec2 getFingerVector(
    int landmark1,
    int landmark2,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords)
{
    // Vec2 is our alias for std::array<double, 2>
    // Curly brace initialization {} is the modern C++ way to initialize values
    return Vec2{
        x_coords[landmark2] - x_coords[landmark1],
        y_coords[landmark2] - y_coords[landmark1]
    };
}

// -----------------------------------------------------------------------------
double getAngleBetweenFingers(
    const Vec2& v1,
    const Vec2& v2)
{
    double dot   = v1[0] * v2[0] + v1[1] * v2[1];
    double norm1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
    double norm2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
    double norms = norm1 * norm2;

    if (norms == 0.0) return 0.0;

    // std::clamp keeps the value between -1 and 1 to avoid NaN from acos
    
    double cosAngle = std::clamp(dot / norms, -1.0, 1.0);
    return toDegrees(std::acos(cosAngle));
}

// -----------------------------------------------------------------------------
double getFingerAngle(
    const Vec2& vector,
    const Vec2& reference_vector)
{
    // Rotation-invariant angle — measures relative to hand baseline
    // not relative to screen horizontal
    return getAngleBetweenFingers(vector, reference_vector);
}

// -----------------------------------------------------------------------------
double getDistancesAggregated(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    double scale)
{
    // Compute mean pairwise distance across all 21 landmarks
    // This replicates scipy.spatial.distance.pdist(...).mean()
    int n = static_cast<int>(x_coords.size());
    // 'static_cast<int>' is C++'s explicit type conversion 

    double sum   = 0.0;
    int    count = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = x_coords[i] - x_coords[j];
            double dy = y_coords[i] - y_coords[j];
            sum += std::sqrt(dx * dx + dy * dy);
            ++count;
            // Pre-increment (++count) is preferred in C++ over post-increment
            // (count++) for non-trivial types as it can be slightly more efficient
        }
    }

    return (count > 0) ? (sum / count) * scale : 0.0;
    // Ternary operator: condition ? value_if_true : value_if_false
    
}

// -----------------------------------------------------------------------------
bool computeFeatures(
    const LandmarkData& landmarks,
    FrameFeatures& out_features)
{
    const auto& x = landmarks.x_coords;
    const auto& y = landmarks.y_coords;
    // 'auto' lets the compiler infer the type 
    // but still checked at compile time. Here it infers const std::vector<double>&

    // Guard — bad frame, skip it
    double palm_size = getLandmarkDistance(9, 0, x, y);
    if (palm_size == 0.0) return false;

    double scale = REFERENCE_PALM_SIZE / palm_size;

    // Compute all direction vectors
    
    Vectors v;
    v.dirV_index       = getFingerVector(7,  6,  x, y);
    v.dirV_middle      = getFingerVector(11, 10, x, y);
    v.dirV_ring        = getFingerVector(15, 14, x, y);
    v.dirV_pinky       = getFingerVector(19, 18, x, y);
    v.dirV_baseline1   = getFingerVector(1,  0,  x, y);
    v.dirV_middle_base = getFingerVector(10, 9,  x, y);
    v.dirV_ring_base   = getFingerVector(14, 13, x, y);

    // Normalize vectors to unit length — removes magnitude, keeps direction
   
    // [](Vec2& vec) { ... } captures nothing, takes a Vec2 by reference
    auto normalize = [](Vec2& vec) {
        double mag = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
        if (mag > 0.0) {
            vec[0] /= mag;
            vec[1] /= mag;
        }
    };

    normalize(v.dirV_index);
    normalize(v.dirV_middle);
    normalize(v.dirV_ring);
    normalize(v.dirV_pinky);
    normalize(v.dirV_baseline1);
    normalize(v.dirV_middle_base);
    normalize(v.dirV_ring_base);

    // Compute angles — all relative to baseline (rotation-invariant)
    Angles a;
    a.angle_index              = getFingerAngle(v.dirV_index,       v.dirV_baseline1);
    a.angle_middle             = getFingerAngle(v.dirV_middle,      v.dirV_baseline1);
    a.angle_ring               = getFingerAngle(v.dirV_ring,        v.dirV_baseline1);
    a.angle_pinky              = getFingerAngle(v.dirV_pinky,       v.dirV_baseline1);
    a.angle_baseline1          = getFingerAngle(v.dirV_baseline1,   v.dirV_baseline1); // always 0
    a.angle_middle_to_ring     = getAngleBetweenFingers(v.dirV_middle,      v.dirV_ring);
    a.angle_base_middle_to_ring= getAngleBetweenFingers(v.dirV_middle_base, v.dirV_ring_base);
    a.angle_middle_to_baseline1= getAngleBetweenFingers(v.dirV_middle,      v.dirV_baseline1);

    // Compute distances — scaled to palm size
    Distances d;
    d.dis_index_tip_to_base  = getLandmarkDistance(8,  0, x, y) * scale;
    d.dis_middle_tip_to_base = getLandmarkDistance(12, 0, x, y) * scale;
    d.dis_ring_tip_to_base   = getLandmarkDistance(16, 0, x, y) * scale;
    d.dis_pinky_tip_to_base  = getLandmarkDistance(20, 0, x, y) * scale;

    // Fill output struct
    out_features.palm_size             = palm_size;
    out_features.scale_factor          = scale;
    out_features.angles                = a;
    out_features.distances             = d;
    out_features.distances_aggregated  = getDistancesAggregated(x, y, scale);
    out_features.vectors               = v;

    return true;
}

// -----------------------------------------------------------------------------
std::vector<double> flattenFeatures(const FrameFeatures& f) {
    // Exact order matches CSV after these drops:
    //   - timestamp, gesture_type         (DROP_COLUMNS)
    //   - all landmark_*_x, landmark_*_y  (startswith "landmark")
    //   - dirV_baseline1_x, dirV_baseline1_y (startswith "dirV_baseline")
    // angle_baseline1 IS included (not explicitly dropped)
    // dirV_baseline1 IS excluded (dropped by startswith "dirV_baseline")
    return {
        // Angles (8 — all included, baseline1 stays)
        f.angles.angle_index,
        f.angles.angle_middle,
        f.angles.angle_ring,
        f.angles.angle_pinky,
        f.angles.angle_baseline1,
        f.angles.angle_middle_to_ring,
        f.angles.angle_base_middle_to_ring,
        f.angles.angle_middle_to_baseline1,
        // Distances (4)
        f.distances.dis_index_tip_to_base,
        f.distances.dis_middle_tip_to_base,
        f.distances.dis_ring_tip_to_base,
        f.distances.dis_pinky_tip_to_base,
        // Aggregated distance (1)
        f.distances_aggregated,
        // Vectors (6 vectors x 2 = 12 — dirV_baseline1 excluded)
        f.vectors.dirV_index[0],       f.vectors.dirV_index[1],
        f.vectors.dirV_middle[0],      f.vectors.dirV_middle[1],
        f.vectors.dirV_ring[0],        f.vectors.dirV_ring[1],
        f.vectors.dirV_pinky[0],       f.vectors.dirV_pinky[1],
        f.vectors.dirV_middle_base[0], f.vectors.dirV_middle_base[1],
        f.vectors.dirV_ring_base[0],   f.vectors.dirV_ring_base[1],
    };
    // Total: 8 + 4 + 1 + 12 = 25 features per frame
    // Full window: 25 x 5 = 125 features
}

} // namespace Features