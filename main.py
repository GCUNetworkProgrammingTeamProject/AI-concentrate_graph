import eyetracking
import emotion
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib import request


path = "~/Desktop/AI-concentrate_graph/"
dat = "shape_predictor_68_face_landmarks.dat"
remote_dat = "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"

def check_dat():
    if(os.path.exists(path + dat)):
        return
    else:
        request.urlretrieve(remote_dat,(path+dat))
        return



if __name__ == "__main__":
    check_dat()
    eyetracking_score = eyetracking.calculate_eyetracking()
    emotion_score = emotion.calculate_emotion()