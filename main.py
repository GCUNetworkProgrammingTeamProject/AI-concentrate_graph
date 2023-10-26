import eyetracking
import emotion
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from urllib import request


path = "/Users/jihyeokchoi/Desktop/AI-concentrate_graph/data/"
dat = "shape_predictor_68_face_landmarks.dat"
remote_dat = "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"


emotion_score, eyetracking_score, sec, frame_count = 0.0, 0.0, 0, 0
concentrate_score = {}


# shape_predictor_68_face_landmarks.dat 파일 없으면 다운로드
# 오류있음 나중에수정
def check_dat():
    if(os.path.exists(path + dat)):
        return
    else:
        request.urlretrieve(remote_dat,(path+dat))
        return


if __name__ == "__main__":
    check_dat()
    
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 얼굴 인식 부분
        num_faces = 0
        frame_count += 1
        if frame_count == 30:
            # 30프레임이 채워지면 1초 가정..?
            frame_count = 0
            sec += 1

            eyetracking_score += eyetracking.calculate_eyetracking(gray)
            emotion_score = emotion.calculate_emotion(gray)
            score = round(eyetracking_score , 4) * 0.8 + round(emotion_score , 4) * 0.2
            score = score * 2 - 1
            concentrate_score[sec] = score
            eyetracking_score, emotion_score = 0.0, 0.0
        else:
            eyetracking_score += eyetracking.calculate_eyetracking(gray)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        # ESC or 'q' 입력시 프로그램 종료
        if key == 27 or key == 113:
            break
    
    # 그래프
    # plt.figure(figsize=[8,6])
    plt.plot(concentrate_score.keys(),concentrate_score.values(),'r',linewidth=2.0)
    plt.legend(['concentrate score'],fontsize=18)
    plt.xlabel('sec',fontsize=16)
    plt.ylabel('score',fontsize=16)
    plt.title('concentration graph',fontsize=16)

    # 종료
    cap.release()
    cv2.destroyAllWindows()