import eyetracking
import emotion
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from urllib import request


# path = "/Users/jihyeokchoi/Desktop/AI-concentrate_graph/data/"
path = "C:\\Users\\jihyeok\\Desktop\\project\\AI-concentrate_graph\\data\\"
dat = "shape_predictor_68_face_landmarks.dat"
remote_dat = "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"


# shape_predictor_68_face_landmarks.dat 파일 없으면 다운로드
# 오류있음 나중에수정
def check_dat():
    if(os.path.exists(path + dat)):
        return
    else:
        request.urlretrieve(remote_dat,(path+dat))
        return

'''
if __name__ == "__main__":
    check_dat()
    
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:\\Users\\jihyeok\\Desktop\\project\\AI-concentrate_graph\\data\\test.mp4")

    while True:
        _, frame = cap.read()
        if(frame is None): # 비디오일때 오류해결
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 얼굴 인식 부분
        num_faces = 0
        frame_count += 1
        if frame_count == 30:
            # 30프레임이 채워지면 1초 가정..?
            frame_count = 0
            sec += 1

            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
            emotion_score = emotion.calculate_emotion(frame)
            score = round(eyetracking_score , 4) * 0.8 + round(emotion_score , 4) * 0.2
            score = score * 2 - 1
            concentrate_score[sec] = score
            print('score: ',score)
            eyetracking_score, emotion_score = 0.0, 0.0
        else:
            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
        
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
    plt.show()

    # 종료
    cap.release()
    cv2.destroyAllWindows()
'''
def concentrate(videoName):
    emotion_score, eyetracking_score, sec, frame_count = 0.0, 0.0, 0, 0
    concentrate_score = {}
    # check_dat()
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("C:\\Users\\jihyeok\\Desktop\\project\\AI-concentrate_graph\\data\\test2.mp4")
    url = "/home/t23209/educrat/public/assests/analysisVideo/" + videoName
    print("경로: ",url)
    cap = cv2.VideoCapture(url)


    while True:
        _, frame = cap.read()
        if(frame is None): # 비디오일때 오류해결
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 얼굴 인식 부분
        num_faces = 0
        frame_count += 1
        if frame_count == 30:
            # 30프레임이 채워지면 1초 가정..?
            frame_count = 0
            sec += 1

            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
            emotion_score = emotion.calculate_emotion(frame)
            score = round(eyetracking_score , 4) * 0.8 + round(emotion_score , 4) * 0.2
            score = score * 2 - 1
            concentrate_score[sec] = score
            print('score: ',score)
            eyetracking_score, emotion_score = 0.0, 0.0
        else:
            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
        

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1)

        # # ESC or 'q' 입력시 프`로그램 종료
        # if key == 27 or key == 113:
        #     break
        
    
    # 그래프
    # plt.figure(figsize=[8,6])
    '''
    plt.plot(concentrate_score.keys(),concentrate_score.values(),'r',linewidth=2.0)
    plt.legend(['concentrate score'],fontsize=18)
    plt.xlabel('sec',fontsize=16)
    plt.ylabel('score',fontsize=16)
    plt.title('concentration graph',fontsize=16)
    plt.show()
    '''

    # 종료
    cap.release()
    cv2.destroyAllWindows()
    return concentrate_score

async def call_concentrate():
    result = await concentrate()
    return result

def concentrate_test():
    emotion_score, eyetracking_score, sec, frame_count = 0.0, 0.0, 0, 0
    concentrate_score = {}
    # check_dat()
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:\\Users\\jihyeok\\Desktop\\project\\AI-concentrate_graph\\data\\test2.mp4")

    while True:
        _, frame = cap.read()
        if(frame is None): # 비디오일때 오류해결
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 얼굴 인식 부분
        num_faces = 0
        frame_count += 1
        if frame_count == 30:
            # 30프레임이 채워지면 1초 가정..?
            frame_count = 0
            sec += 1

            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
            emotion_score = emotion.calculate_emotion(frame)
            score = round(eyetracking_score , 4) * 0.8 + round(emotion_score , 4) * 0.2
            score = score * 2 - 1
            concentrate_score[sec] = score
            print('score: ',score)
            eyetracking_score, emotion_score = 0.0, 0.0
        else:
            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
        

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        # ESC or 'q' 입력시 프`로그램 종료
        if key == 27 or key == 113:
            break
        
    
    # 그래프
    # plt.figure(figsize=[8,6])
    '''
    plt.plot(concentrate_score.keys(),concentrate_score.values(),'r',linewidth=2.0)
    plt.legend(['concentrate score'],fontsize=18)
    plt.xlabel('sec',fontsize=16)
    plt.ylabel('score',fontsize=16)
    plt.title('concentration graph',fontsize=16)
    plt.show()
    '''

    # 종료
    cap.release()
    cv2.destroyAllWindows()
    return concentrate_score




if __name__ == "__main__":
    emotion_score, eyetracking_score, sec, frame_count = 0.0, 0.0, 0, 0
    concentrate_score = {}
    # check_dat()
    
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:\\Users\\jihyeok\\Desktop\\project\\AI-concentrate_graph\\data\\test.mp4")

    while True:
        _, frame = cap.read()
        if(frame is None): # 비디오일때 오류해결
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 얼굴 인식 부분
        num_faces = 0
        frame_count += 1
        if frame_count == 30:
            # 30프레임이 채워지면 1초 가정..?
            frame_count = 0
            sec += 1

            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
            emotion_score = emotion.calculate_emotion(frame)
            score = round(eyetracking_score , 4) * 0.8 + round(emotion_score , 4) * 0.2
            score = score * 2 - 1
            concentrate_score[sec] = score
            print('score: ',score)
            eyetracking_score, emotion_score = 0.0, 0.0
        else:
            eyetracking_score += eyetracking.calculate_eyetracking(gray, frame)
            continue
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        # ESC or 'q' 입력시 프`로그램 종료
        if key == 27 or key == 113:
            break
    
    # 그래프
    # plt.figure(figsize=[8,6])
    plt.plot(concentrate_score.keys(),concentrate_score.values(),'r',linewidth=2.0)
    plt.legend(['concentrate score'],fontsize=18)
    plt.xlabel('sec',fontsize=16)
    plt.ylabel('score',fontsize=16)
    plt.title('concentration graph',fontsize=16)
    plt.show()

    # 종료
    cap.release()
    cv2.destroyAllWindows()
