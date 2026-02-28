"""
Landmark Server
===============
Reads webcam frames, extracts hand landmarks via MediaPipe,
and sends them to the C++ process over a Unix domain socket.

Sends per frame:
    - 1 byte:  hand detected flag (1 = detected, 0 = not detected)
    - If detected: 42 doubles (21 x coords + 21 y coords) = 336 bytes

Usage:
    python landmark_server.py
    (start this BEFORE running ./rps_cpp)
"""

import socket
import struct
import os
import time
import cv2
import mediapipe as mp
from rps_mediapipe_landmarks_ML import HandGestureRecognizer

# =============================================================================
# CONFIGURATION
# =============================================================================

SOCKET_PATH = "/tmp/rps_landmarks.sock"  # Unix socket file path
FPS_DELAY   = 0.033                      # ~30fps

# =============================================================================
# MAIN
# =============================================================================

def run_server():
    # Remove stale socket file if it exists from a previous run
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    # Create Unix domain socket
    # AF_UNIX = Unix domain socket (local only, faster than TCP)
    # SOCK_STREAM = reliable ordered byte stream (like TCP but local)
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(SOCKET_PATH)
    server_sock.listen(1)  # accept 1 pending connection

    print(f"Landmark server listening on: {SOCKET_PATH}")
    print("Waiting for C++ client to connect...")

    # Block until C++ connects
    conn, _ = server_sock.accept()
    print("C++ client connected. Starting landmark streaming...")

    recognizer = HandGestureRecognizer(data_collection_mode=False)

    try:
        recognizer.initialize()

        while True:
            ret, frame = recognizer.cap.read()
            if not ret:
                print("Failed to read from webcam")
                break

            # Send frame to MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            recognizer.landmarker.detect_async(mp_image, timestamp_ms)

            if recognizer.is_hand_detected():
                data = recognizer.get_latest_data()
                if data:
                    x_coords = data['x_coords']  # list of 21 floats
                    y_coords = data['y_coords']   # list of 21 floats

                    # Pack as binary:
                    # '!' = network byte order (big-endian, standard)
                    # 'B' = 1 unsigned byte (hand detected flag)
                    # '42d' = 42 doubles (8 bytes each = 336 bytes total)
                    payload = struct.pack('!B42d', 1, *x_coords, *y_coords)
                    try:
                        conn.sendall(payload)
                    except BrokenPipeError:
                        print("C++ client disconnected")
                        break
            else:
                # Send "no hand" signal — just 1 byte
                try:
                    conn.sendall(struct.pack('!B', 0))
                except BrokenPipeError:
                    print("C++ client disconnected")
                    break

            time.sleep(FPS_DELAY)

    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        conn.close()
        server_sock.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        recognizer.cleanup()
        print("Landmark server shut down")


if __name__ == "__main__":
    run_server()