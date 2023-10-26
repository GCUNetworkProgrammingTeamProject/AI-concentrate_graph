#import warnings
#warnings.filterwarnings('ignore')

# 데이터 확인
import numpy as np
import matplotlib.pyplot as plt

# Detect Face
import cv2
from scipy.ndimage import zoom

# Model
from keras.models import load_model

shape_x = 48
shape_y = 48

# 전체 이미지에서 얼굴을 찾아내는 함수
def detect_face(frame):
    
    # cascade pre-trained 모델 불러오기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # RGB를 gray scale로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cascade 멀티스케일 분류
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor = 1.1,
                                                   minNeighbors = 6,
                                                   minSize = (shape_x, shape_y),
                                                   flags = cv2.CASCADE_SCALE_IMAGE
                                                  )
    
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y+h, x:x+w]
            coord.append([x, y, w, h])
            
    return gray, detected_faces, coord
#print(cv2.data.haarcascades)

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수
def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:
        
        # 얼굴로 감지된 영역
        x, y, w, h = det
        
        # 이미지 경계값 받기
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
        
        # gray scacle 에서 해당 위치 가져오기
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        
        # 얼굴 이미지만 확대
        new_extracted_face = zoom(extracted_face, (shape_x/extracted_face.shape[0], shape_y/extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) # sacled
        new_face.append(new_extracted_face)
        
    return new_face

suzy = cv2.imread('dataset/image.jpeg')
plt.imshow(cv2.cvtColor(suzy, cv2.COLOR_BGR2RGB))

def calculate_emotion(frame):
    model = load_model('emotion_recognition.h5')
    
    # 얼굴 추출
    gray, detected_faces, coord = detect_face(frame)
    face_zoom = extract_face_features(gray, detected_faces, coord)

    # 모델 추론
    input_data = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
    output_data = model.predict(input_data)
    
    '''
    emotion_array = ['angry','disgust','fear','happy','sad','surprise','neutral'];

    for i in range(output_data.shape[1]):
        print(f'{emotion_array[i]}: {(output_data[0][i] / sum(output_data[0]))*100}')
    '''

    emotion_score = round(output_data[0][-1] / sum(output_data[0]), 4)
    return emotion_score