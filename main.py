import eyetracking
import emotion
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    eyetracking_score = eyetracking.calculate_eyetracking()
    emotion_score = emotion.calculate_emotion()