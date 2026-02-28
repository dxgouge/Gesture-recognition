#include "capture.h"

#include <iostream>
#include <stdexcept>
#include <cstring>    // memcpy

// POSIX socket headers — Unix-specific networking API
// These don't exist on Windows — one reason C++ networking is platform-specific
#include <sys/socket.h>  // socket(), connect(), recv()
#include <sys/un.h>      // sockaddr_un — Unix domain socket address struct
#include <unistd.h>      // close()
#include <arpa/inet.h>   // ntohl(), used for byte order conversion

namespace Capture {

// =============================================================================
// COLOR HELPER
// =============================================================================

cv::Scalar getGestureColor(const std::string& gesture_name) {
    if      (gesture_name == "Rock")     return COLOR_ROCK;
    else if (gesture_name == "Paper")    return COLOR_PAPER;
    else if (gesture_name == "Scissors") return COLOR_SCISSORS;
    else                                 return COLOR_UNKNOWN;
}

// =============================================================================
// DRAW OVERLAY
// =============================================================================

void drawOverlay(
    cv::Mat& frame,
    const std::string& gesture_name,
    double confidence,
    int buffer_size,
    int window_size)
{
    cv::Scalar color = getGestureColor(gesture_name);

    cv::rectangle(frame, cv::Point(10, 10), cv::Point(320, 90),
                  cv::Scalar(20, 20, 20), -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(320, 90),
                  color, 2);

    cv::putText(frame, gesture_name,
                cv::Point(20, 58),
                cv::FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv::LINE_AA);

    std::string conf_str = std::to_string(static_cast<int>(confidence * 100)) + "%";
    cv::putText(frame, conf_str,
                cv::Point(210, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);

    int bar_w = static_cast<int>(
        (static_cast<double>(buffer_size) / window_size) * 290
    );
    cv::rectangle(frame, cv::Point(10, 75), cv::Point(300, 85),
                  cv::Scalar(40, 40, 40), -1);
    if (bar_w > 0) {
        cv::rectangle(frame, cv::Point(10, 75), cv::Point(10 + bar_w, 85),
                      color, -1);
    }

    std::string buf_str = "Buffer " + std::to_string(buffer_size)
                        + "/" + std::to_string(window_size);
    cv::putText(frame, buf_str,
                cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(160, 160, 160), 1);
}

// =============================================================================
// SOCKET: Connect to Python server
// Returns socket file descriptor — an integer handle the OS gives us
// Like Python's conn = socket.socket(); conn.connect(path)
// =============================================================================

int connectToServer(const std::string& socket_path) {
    // Create a Unix domain socket
    // AF_UNIX = local socket, SOCK_STREAM = reliable stream
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    // sockaddr_un is the address structure for Unix domain sockets
    // It just holds the file path — much simpler than TCP addresses
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));           // zero out the struct
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    std::cout << "Connecting to landmark server at: " << socket_path << std::endl;
    std::cout << "Make sure landmark_server.py is running first." << std::endl;

    // connect() blocks until the server accepts
    if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock_fd);
        throw std::runtime_error(
            "Could not connect to landmark server. Is landmark_server.py running?"
        );
    }

    std::cout << "Connected to landmark server." << std::endl;
    return sock_fd;
}

// =============================================================================
// SOCKET: Receive one landmark message
// Protocol mirrors what Python's struct.pack sends:
//   1 byte  — hand detected flag (0 or 1)
//   336 bytes — 42 doubles if hand detected (skipped if not)
// =============================================================================

bool receiveLandmarks(int socket_fd, LandmarkMessage& out_msg) {
    // Step 1: Read the 1-byte hand detected flag
    uint8_t flag = 0;
    // recv() is the C socket receive function
    // MSG_WAITALL means block until all requested bytes arrive
    ssize_t n = recv(socket_fd, &flag, sizeof(flag), MSG_WAITALL);
    if (n <= 0) {
        // n == 0 means server closed connection, n < 0 means error
        return false;
    }

    out_msg.hand_detected = (flag == 1);

    if (!out_msg.hand_detected) {
        // No hand — nothing more to read for this frame
        return true;
    }

    // Step 2: Read 42 doubles (336 bytes)
    // Python's struct.pack('!42d') sends them in network byte order (big-endian)
    // Most Macs are little-endian so we need to swap bytes
    constexpr size_t DATA_SIZE = COORDS_PER_MSG * sizeof(double); // 336 bytes
    double buffer[COORDS_PER_MSG];

    n = recv(socket_fd, buffer, DATA_SIZE, MSG_WAITALL);
    if (n != static_cast<ssize_t>(DATA_SIZE)) {
        return false;
    }

    // Convert from network byte order (big-endian) to host byte order
    // doubles don't have a standard ntoh function so we swap manually
    // This is the kind of low-level work Python's struct module hides from you
    for (int i = 0; i < COORDS_PER_MSG; ++i) {
        uint64_t raw;
        memcpy(&raw, &buffer[i], sizeof(uint64_t));

        // __builtin_bswap64 swaps all 8 bytes — compiler intrinsic, very fast
        raw = __builtin_bswap64(raw);
        memcpy(&buffer[i], &raw, sizeof(double));
    }

    // Split into x and y arrays
    for (int i = 0; i < NUM_LANDMARKS; ++i) {
        out_msg.x_coords[i] = buffer[i];
        out_msg.y_coords[i] = buffer[NUM_LANDMARKS + i];
    }

    return true;
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void runLoop(Inference::RockPaperScissorsClassifier& classifier) {
    // Connect to Python landmark server
    int sock_fd = connectToServer(SOCKET_PATH);

    // Open webcam — C++ side only for display
    // Python controls the MediaPipe camera, we open a second handle for display
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        close(sock_fd);
        throw std::runtime_error("Could not open webcam for display");
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    std::cout << "Press 'q' or ESC to quit." << std::endl;

    std::deque<std::vector<double>> frame_buffer;
    std::string current_gesture    = "Unknown";
    double      current_confidence = 0.0;
    int         frame_count        = 0;
    cv::Mat     frame;

    while (true) {
        // Grab display frame
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);

        frame_count++;

        // Receive landmarks from Python
        LandmarkMessage msg;
        if (!receiveLandmarks(sock_fd, msg)) {
            std::cerr << "Lost connection to landmark server" << std::endl;
            break;
        }

        if (msg.hand_detected) {
            // Build LandmarkData from received coords
            Features::LandmarkData landmarks;
            landmarks.x_coords.assign(msg.x_coords, msg.x_coords + NUM_LANDMARKS);
            landmarks.y_coords.assign(msg.y_coords, msg.y_coords + NUM_LANDMARKS);
            // .assign(begin_ptr, end_ptr) fills a vector from a raw array
            // Like Python's list(array)

            Features::FrameFeatures frame_features;
            bool ok = Features::computeFeatures(landmarks, frame_features);

            if (ok) {
                std::vector<double> flat = Features::flattenFeatures(frame_features);
                frame_buffer.push_back(flat);

                while (static_cast<int>(frame_buffer.size()) > Inference::WINDOW_SIZE) {
                    frame_buffer.pop_front();
                }

                if (static_cast<int>(frame_buffer.size()) == Inference::WINDOW_SIZE &&
                    frame_count % PREDICT_EVERY_N == 0)
                {
                    std::vector<double> window;
                    window.reserve(Inference::WINDOW_SIZE * flat.size());
                    for (const auto& f : frame_buffer) {
                        window.insert(window.end(), f.begin(), f.end());
                    }

                    try {
                        Inference::PredictionResult result = classifier.predict(window);
                        current_gesture    = result.gesture_name;
                        current_confidence = result.confidence;
                    } catch (const std::exception& e) {
                        std::cerr << "Prediction error: " << e.what() << std::endl;
                    }
                }
            }
        } else {
            frame_buffer.clear();
            current_gesture    = "Unknown";
            current_confidence = 0.0;
        }

        drawOverlay(frame, current_gesture, current_confidence,
                    static_cast<int>(frame_buffer.size()), Inference::WINDOW_SIZE);

        cv::imshow(WINDOW_NAME, frame);

        int key = cv::waitKey(FRAME_DELAY_MS) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    close(sock_fd);
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Shutting down." << std::endl;
}

} // namespace Capture