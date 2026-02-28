cmake_minimum_required(VERSION 3.16)
project(rps_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# =============================================================================
# FIND OPENCV
# find_package(OpenCV) is the standard CMake way to locate OpenCV.
# It sets OpenCV_INCLUDE_DIRS and OpenCV_LIBS automatically.
# We hint the path for Homebrew in case CMake can't find it on its own.
# =============================================================================
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# =============================================================================
# FIND LIGHTGBM
# =============================================================================
find_package(LightGBM QUIET)

if(LightGBM_FOUND)
    message(STATUS "Found LightGBM via find_package")
else()
    message(STATUS "Using Homebrew LightGBM paths")
    if(EXISTS "/opt/homebrew/opt/lightgbm/include/LightGBM")
        set(LIGHTGBM_INCLUDE_DIR "/opt/homebrew/opt/lightgbm/include")
        set(LIGHTGBM_LIB_DIR     "/opt/homebrew/opt/lightgbm/lib")
    elseif(EXISTS "/usr/local/opt/lightgbm/include/LightGBM")
        set(LIGHTGBM_INCLUDE_DIR "/usr/local/opt/lightgbm/include")
        set(LIGHTGBM_LIB_DIR     "/usr/local/opt/lightgbm/lib")
    else()
        message(FATAL_ERROR "Could not find LightGBM. Run: brew install lightgbm")
    endif()
endif()

# =============================================================================
# EXECUTABLE
# =============================================================================
add_executable(rps_cpp
    src/main.cpp
    src/features.cpp
    src/inference.cpp
    src/capture.cpp
)

target_include_directories(rps_cpp PRIVATE
    src
    ${OpenCV_INCLUDE_DIRS}
    ${LIGHTGBM_INCLUDE_DIR}
)

target_link_directories(rps_cpp PRIVATE
    ${LIGHTGBM_LIB_DIR}
)

target_link_libraries(rps_cpp PRIVATE
    ${OpenCV_LIBS}   # all OpenCV modules
    _lightgbm
)