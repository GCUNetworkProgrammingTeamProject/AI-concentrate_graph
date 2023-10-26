import cv2
import numpy as np
import dlib
from math import hypot
import face_recognition
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt


""" global variables
"""
num_frames = 0
short_cheating_count = 0
long_cheating_count = 0

#집중도 변수
eyetracking_score = 0
wrong_eye_direction = False
wrong_head_direction = False

is_time_counting_eye = False    # 눈동자 이탈 시간 측정용
is_time_counting_head = False   # 고개 이탈 시간 측정용

start_time_eye = 0
start_time_head = 0
#기준 설정할 프레임
criteria_frame_num = 5
warning_count = 1
cause = 0      # 1 : 눈동자 오른쪽
               # 2 : 눈동자 왼쪽
               # 3 : 고개 오른쪽
               # 4 : 고개 왼쪽
               # 5 : 얼굴 안보임
               # 6 : 신원인증 결과값
               # 7 : 얼굴개수 여러개

no_face_time = 2       # 몇초간 얼굴 탐지 안될 시 경고할지
max_short_cheating = 5  # 짧은 시간 부정행위 횟수
max_long_cheating = 1   # 짧은 시간 부정행위 횟수
criteria_time = 5       # 짧은 시간 긴 시간 나누는 기준초

path = "C:/Users/takea/Desktop/NetworkProgramming/codes/AI/eyetracking/"
path_identification = "C:/Users/takea/Desktop/NetworkProgramming/codes/AI/eyetracking/"
filename = "video_"

frame_count = 0

""" determine mid point
"""
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)



""" Detect face & eye's location
    and blinking
"""
def get_blinking_ratio(facial_landmarks):
    left_point1 = (facial_landmarks.part(36).x, facial_landmarks.part(36).y)
    right_point1 = (facial_landmarks.part(39).x, facial_landmarks.part(39).y)
    center_top1 = midpoint(facial_landmarks.part(37), facial_landmarks.part(38))
    center_bottom1 = midpoint(facial_landmarks.part(41), facial_landmarks.part(40))

    left_point2 = (facial_landmarks.part(42).x, facial_landmarks.part(42).y)
    right_point2 = (facial_landmarks.part(45).x, facial_landmarks.part(45).y)
    center_top2 = midpoint(facial_landmarks.part(43), facial_landmarks.part(44))
    center_bottom2 = midpoint(facial_landmarks.part(47), facial_landmarks.part(46))

    ver_line_len1 = hypot((center_top1[0] - center_bottom1[0]), (center_top1[1] - center_bottom1[1]))
    hor_line_len1 = hypot((left_point1[0] - right_point1[0]), (left_point1[1] - right_point1[1]))
    ver_line_len2 = hypot((center_top2[0] - center_bottom2[0]), (center_top2[1] - center_bottom2[1]))
    hor_line_len2 = hypot((left_point2[0] - right_point2[0]), (left_point2[1] - right_point2[1]))

    blink_ratio_left = hor_line_len1 / ver_line_len1
    blink_ratio_right = hor_line_len2 / ver_line_len2
    blink_ratio = (blink_ratio_left + blink_ratio_right) / 2

    return blink_ratio



"""Print face's area
"""
def print_face(facial_landmarks, _gray, _frame):
    face_region = np.array([(facial_landmarks.part(0).x, facial_landmarks.part(0).y),
                            (facial_landmarks.part(1).x, facial_landmarks.part(1).y),
                            (facial_landmarks.part(2).x, facial_landmarks.part(2).y),
                            (facial_landmarks.part(3).x, facial_landmarks.part(3).y),
                            (facial_landmarks.part(4).x, facial_landmarks.part(4).y),
                            (facial_landmarks.part(5).x, facial_landmarks.part(5).y),
                            (facial_landmarks.part(6).x, facial_landmarks.part(6).y),
                            (facial_landmarks.part(7).x, facial_landmarks.part(7).y),
                            (facial_landmarks.part(8).x, facial_landmarks.part(8).y),
                            (facial_landmarks.part(9).x, facial_landmarks.part(9).y),
                            (facial_landmarks.part(10).x, facial_landmarks.part(10).y),
                            (facial_landmarks.part(11).x, facial_landmarks.part(11).y),
                            (facial_landmarks.part(12).x, facial_landmarks.part(12).y),
                            (facial_landmarks.part(13).x, facial_landmarks.part(13).y),
                            (facial_landmarks.part(14).x, facial_landmarks.part(14).y),
                            (facial_landmarks.part(15).x, facial_landmarks.part(15).y),
                            (facial_landmarks.part(16).x, facial_landmarks.part(16).y),
                            (facial_landmarks.part(18).x, facial_landmarks.part(18).y),
                            (facial_landmarks.part(23).x, facial_landmarks.part(23).y)], np.int32)

    cv2.polylines(_frame, [face_region], True, (0, 255, 255), 1)



""" Detect eye's gazing
"""
def get_gaze_ratio(eye_points, facial_landmarks, _gray, _frame):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    cv2.polylines(_frame, [eye_region], True, (0, 255, 255), 1)

    height, width, _ = _frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 1)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(_gray, _gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # 눈동자의 흰부분 계산으로 보는 방향 추정
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    #cv2.imshow("left", left_side_threshold)
    #cv2.imshow("right", right_side_threshold)

    # left, right side white 가 10 미만이면 눈 감은것으로 인식
    if left_side_white < 5 or right_side_white < 5:
        _gaze_ratio = 1
    else:
        _gaze_ratio = left_side_white / right_side_white

    return _gaze_ratio



""" Detect head's direction
"""
def get_head_angle_ratio(head_points, facial_landmarks, _frame):
    # 코의 가로선 표시
    nose_region1 = np.array([(facial_landmarks.part(head_points[0]).x, facial_landmarks.part(head_points[0]).y),
                             (facial_landmarks.part(head_points[1]).x, facial_landmarks.part(head_points[1]).y),
                             (facial_landmarks.part(head_points[2]).x, facial_landmarks.part(head_points[2]).y),
                             (facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y)],
                            np.int32)
    cv2.polylines(_frame, [nose_region1], True, (0, 255, 255), 1)

    # 코의 세로선 표시
    nose_region2 = np.array([(facial_landmarks.part(head_points[4]).x, facial_landmarks.part(head_points[4]).y),
                             (facial_landmarks.part(head_points[5]).x, facial_landmarks.part(head_points[5]).y),
                             (facial_landmarks.part(head_points[6]).x, facial_landmarks.part(head_points[6]).y),
                             (facial_landmarks.part(head_points[7]).x, facial_landmarks.part(head_points[7]).y),
                             (facial_landmarks.part(head_points[8]).x, facial_landmarks.part(head_points[8]).y)],
                            np.int32)
    cv2.polylines(_frame, [nose_region2], True, (0, 255, 255), 1)

    # 코의 왼쪽 기준선 표시
    nose_line_left = np.array([(facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y),
                               (facial_landmarks.part(head_points[4]).x, facial_landmarks.part(head_points[4]).y)],
                              np.int32)
    cv2.polylines(_frame, [nose_line_left], True, (255, 0, 255), 1)

    # 코의 오른쪽 기준선 표시
    nose_line_right = np.array([(facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y),
                                (facial_landmarks.part(head_points[8]).x, facial_landmarks.part(head_points[8]).y)],
                               np.int32)
    cv2.polylines(_frame, [nose_line_right], True, (255, 0, 255), 1)

    nose_left_point = (facial_landmarks.part(head_points[4]).x, facial_landmarks.part(head_points[4]).y)
    nose_right_point = (facial_landmarks.part(head_points[8]).x, facial_landmarks.part(head_points[8]).y)
    nose_center_point = (facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y)

    # 오른쪽 기준선과 왼쪽 기준선 길이 계산
    nose_line_len1 = hypot(nose_left_point[0] - nose_center_point[0], nose_left_point[1] - nose_center_point[1])
    nose_line_len2 = hypot(nose_right_point[0] - nose_center_point[0], nose_right_point[1] - nose_center_point[1])

    if nose_line_len1 > nose_line_len2:
        _head_direction = "right"
        _direction_ratio = nose_line_len1 / nose_line_len2
    else:
        _head_direction = "left"
        _direction_ratio = nose_line_len2 / nose_line_len1

    return _head_direction, _direction_ratio

""" Set criteria
"""
def set_criteria(_direction, _head_direction_sum, _criteria_finished, _direction_ratio, _eye_direction_sum):
    global criteria_frame_num, num_frames

    _head_direction_criteria = 0
    _eye_direction_criteria = 0

    num_frames += 1
    print(f"num_frames : {num_frames}")
    if _direction == "left" and (not _criteria_finished):
        _head_direction_sum += (_direction_ratio - 1) * (-1)
        print(f"head_direction_sum : {_head_direction_sum}")
    elif _direction == "right" and (not _criteria_finished):
        _head_direction_sum += (_direction_ratio - 1)
        print(f"head_direction_sum : {_head_direction_sum}")

    if num_frames == criteria_frame_num:
        _head_direction_criteria = (_head_direction_sum / num_frames)
        print("HEAD : {}".format(_head_direction_criteria))
        print(f"set_criteria : True")
        _criteria_finished = True

    if num_frames == criteria_frame_num:
        _eye_direction_criteria = (_eye_direction_sum / num_frames)
        print("EYE : {}".format(_eye_direction_criteria))

    return _head_direction_criteria, _eye_direction_criteria, _criteria_finished, num_frames



def warn_eye_direction(_criteria_finished, _gaze_ratio, _eye_direction_criteria, _margin_eye):

    global is_time_counting_eye, start_time_eye, cause, wrong_eye_direction

    #    숫자가 작아질수록 관대
    if _criteria_finished and _gaze_ratio < _eye_direction_criteria - _margin_eye:
        wrong_eye_direction = True

    #    숫자가 커질수록 관대
    elif _criteria_finished and _gaze_ratio > _eye_direction_criteria + _margin_eye:
        wrong_eye_direction = True

    else:
        wrong_eye_direction = False


""" Head angle warning algorithm
"""
def warn_head_direction(_criteria_finished, _head_direction_criteria, _head_direction, _margin_head):
    global start_time_head, is_time_counting_head, cause, wrong_head_direction

    # 왼쪽 바라볼때
    if _criteria_finished and _head_direction_criteria < 0:
        if _head_direction[0] == "left" and _head_direction[1] > 1 - _head_direction_criteria + _margin_head:
            wrong_head_direction = True

        elif _head_direction[0] == "right" and _head_direction[1] > 1 + _head_direction_criteria + _margin_head:
            wrong_head_direction = True
        else:
            wrong_head_direction = False

    # 오른쪽 바라볼때
    if _criteria_finished and _head_direction_criteria >= 0:
        if _head_direction[0] == "left" and _head_direction[1] > 1 - _head_direction_criteria + _margin_head:
            wrong_head_direction = True

        elif _head_direction[0] == "right" and _head_direction[1] > 1 + _head_direction_criteria + _margin_head:
            wrong_head_direction = True

        else:
            wrong_head_direction = False    


""" count the cheating frequency
"""
def count_cheating(_duration, _cause):
    global short_cheating_count, long_cheating_count, warning_count, \
        max_long_cheating, max_short_cheating, criteria_time, filename, path, eyetracking_score 

    if _duration < criteria_time:
        short_cheating_count += 1

    elif _duration >= criteria_time:
        long_cheating_count += 1


    if short_cheating_count == max_short_cheating:
        s = datetime.now().strftime("%Y%m%d%H%M%S")
        f = open(path + filename + s + ".txt", 'w')
        warning_count += 1
        f.write("짧은 경고 5회 누적 + " + str(_cause))
        f.close()
        print("짧은 경고 5회 누적")
        short_cheating_count = 0

    if long_cheating_count == max_long_cheating:
        print("긴 경고 1회 누적!")
        s = datetime.now().strftime("%Y%m%d%H%M%S")
        f = open(path + filename + s + ".txt", 'w')
        warning_count += 1
        f.write("긴 경고 1회 누적 + " + str(_cause))
        f.close()
        long_cheating_count = 0


""" detect no faces warning
"""
def warn_no_face(_duration):
    global no_face_time, max_long_cheating, max_short_cheating, warning_count, filename, path

    if _duration > no_face_time:
        print("얼굴 감지 안됨 경고")
        s = datetime.now().strftime("%Y%m%d%H%M%S")
        f = open(path + filename + s + ".txt", 'w')
        warning_count += 1
        f.write("{:.1f} 초간 얼굴 감지 안됨 경고 + 5".format(_duration))
        f.close()
        print("{:.1f} 초간 얼굴 감지 안됨 경고 + 5".format(_duration))
        print("얼굴 감지 안됨 !!!")


"""""""""""""""""""""""""""""""""""""""
""""""     # MAIN FUNCTION #     """"""
"""""""""""""""""""""""""""""""""""""""

def main():

    global num_frames, frame_count, eyetracking_score, wrong_eye_direction, wrong_head_direction
    
    wrong_eye_direction = False
    wrong_head_direction = False
    eyetracking_score = 0
    frame_count = 0

    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/takea/anaconda3/envs/networkProgramming/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat")

    # 얼굴 인식 활성화 여부 확인
    Activated = False

    # 얼굴 인식 위한 변수
    start_time_face = 0

    # 초반 고개/눈 방향 기준 설정 위한 변수들
    head_direction_sum = 0
    eye_direction_sum = 0
    head_direction_criteria = 0
    eye_direction_criteria = 0

    # 초반 고개/눈 방향 기준 설정 여부 확인
    criteria_finished = False

    # 눈 방향 탐지 위한 마진 (낮을수록 엄격하게 탐지)
    margin_eye = 0.9
    margin_head = 0.5

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        num_faces = 0
        # 얼굴 인식 부분


    
        frame_count += 1
        if frame_count == 30:
            # 30프레임이 채워지면 평균 집중도를 계산하고 초기화
            frame_count = 0
            eyetracking_score = round(eyetracking_score, 4)
            print(f"concentration_score : {eyetracking_score}")
            eyetracking_score = 0

        if faces:
            eyetracking_score += 1/30
        else:
            eyetracking_score -= 1/30
        for face in faces:
            #print("실행됨")
            num_faces += 1
            landmarks = predictor(gray, face)

            # 일정 시간 이상 얼굴 탐지 안되면 경고
            duration = time.time() - start_time_face
            start_time_face = time.time()
            if not num_frames == 0:
                warn_no_face(duration)

            # 보는 방향 감지
            # 오른쪽 -> 커진다 / 왼쪽 -> 작아진다
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame)
            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2
            eye_direction_sum += gaze_ratio

            # 고개 돌리는 방향 감지
            head_direction = get_head_angle_ratio([27, 28, 29, 30, 31, 32, 33, 34, 35], landmarks, frame)
            direction = head_direction[0]
            direction_ratio = head_direction[1]

            """# 눈을 추적하며 깜박임 감지
            if get_blinking_ratio(landmarks) > 3.7:  # 숫자가 높아질수록 엄격하게 감지
                print("눈 깜빡임")"""

            # 눈동자가 인가된 범위를 벗어나면 경고
            warn_eye_direction(criteria_finished, gaze_ratio, eye_direction_criteria, margin_eye)

            # 고개가 인가된 범위를 벗어나면 경고
            warn_head_direction(criteria_finished, head_direction_criteria, head_direction, margin_head)


            # 최초 100프레임동안 고개, 눈동자 기준설정
            if not criteria_finished:
                print("기준 설정")
                head_direction_criteria, eye_direction_criteria, criteria_finished, num_frames \
                    = set_criteria(direction, head_direction_sum,
                                   criteria_finished, direction_ratio, eye_direction_sum)


            if (wrong_eye_direction or wrong_head_direction):
                eyetracking_score -= 2/30
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        '''
        # Print ont the screen
        if temp_faces_for_compare[2]:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(1)
            cv2.destroyAllWindows()
        '''

        # ESC 입력시 프로그램 종료
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   main()

# 바로 신원인식하면 에러뜸
