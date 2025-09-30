
import cv2
import numpy as np
import time
import os

import os
from pynput import keyboard
import pandas as pd
import mediapipe as mp

import matplotlib.pyplot as plt

csv_file = "C:/Users/dxg45/yolofn/MPMLDATA/rps_mediapipe_landmarks_ML_data.csv"
columns = ["landmark_0_x", "landmark_0_y", "landmark_1_x", "landmark_1_y", "landmark_2_x", "landmark_2_y", "landmark_3_x", "landmark_3_y", "landmark_4_x", "landmark_4_y", "landmark_5_x", "landmark_5_y", "landmark_6_x", "landmark_6_y", "landmark_7_x", "landmark_7_y", "landmark_8_x", "landmark_8_y", "landmark_9_x", "landmark_9_y", "landmark_10_x", "landmark_10_y", "landmark_11_x", "landmark_11_y", "landmark_12_x", "landmark_12_y", "landmark_13_x", "landmark_13_y", "landmark_14_x", "landmark_14_y", "landmark_15_x", "landmark_15_y", "landmark_16_x", "landmark_16_y", "landmark_17_x", "landmark_17_y", "landmark_18_x", "landmark_18_y", "landmark_19_x", "landmark_19_y", "landmark_20_x", "landmark_20_y"]
df = pd.DataFrame(columns)
df = df.to_csv(csv_file, index=False)


