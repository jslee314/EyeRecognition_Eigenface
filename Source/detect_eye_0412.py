#! /usr/bin/python
#import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import os
import glob
# cp /media/sf_ShareFolder/ljs3.mp4 .


def eye_aspect_ratio(eye):
    # image ear detection
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    eye_ear = (a + b) / (2.0 * c)

    return eye_ear


def variance_of_laplacian(image):
    # image blur detection
    # return the focus measure, which is simply the variance of the Laplacian

    return cv2.Laplacian(image, cv2.CV_64F).var()


# 1. constants
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

# 2. dlib의 얼굴 검출기와 얼굴 표식 탐지기를 초기화 (dlib)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3. 좌/우 눈의 좌표값 가져오기 (array slice index values)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lBStart, lBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rBStart, rBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]


# 5. 비디오스트림 시작.

# list_name = ['jwt', 'kti', 'kyh', 'ljs', 'phh', 'ldh', 'nhj']
#list_name = glob.glob("./data/*.mp4")
list_name = glob.glob("/media/sf_ShareFolder/data/mp4/rotation/*.mp4")
print(list_name)
for num_name in range(len(list_name)):
    name = list_name[num_name]
    filename = name

    # 저장 할 디렉토리 생성
    vs = FileVideoStream(filename).start()
    fileStream = True
    time.sleep(1.0)

    os.makedirs('/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/0-20')
    os.makedirs('/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/0-20')
    os.makedirs('/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/20-25')
    os.makedirs('/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/20-25')
    os.makedirs('/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/25-30')
    os.makedirs('/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/25-30')
    os.makedirs('/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/30')
    os.makedirs('/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/30')
    '''

    os.makedirs('/home/hongbog/gen_data/left/' + str(num_name) + '/0-20')
    os.makedirs('/home/hongbog/gen_data/right/' + str(num_name) + '/0-20')
    os.makedirs('/home/hongbog/gen_data/left/' + str(num_name) + '/20-25')
    os.makedirs('/home/hongbog/gen_data/right/' + str(num_name) + '/20-25')
    os.makedirs('/home/hongbog/gen_data/left/' + str(num_name) + '/25-30')
    os.makedirs('/home/hongbog/gen_data/right/' + str(num_name) + '/25-30')
    os.makedirs('/home/hongbog/gen_data/left/' + str(num_name) + '/30')
    os.makedirs('/home/hongbog/gen_data/right/' + str(num_name) + '/30')
    '''
    frame_cnt = 1

    # output data
    data = []
    cnt_30 = 0
    cnt_25 = 0
    cnt_20 = 0
    cnt_00 = 0
    # 5. 비디오 스트림 반복 루프------------------------------------------------
    while True:
        # 5-1 비디오에 프레임이 없으면 루프 탈출.
        if fileStream and not vs.more():
            break
        # 5-2 스트림에서 프레임을 읽은 다음, 크기를 조정하고, 회색 음영으로 변환.
        frame = vs.read()
        frame = imutils.resize(frame, width=720, height=1280)  # inter=cv2.INTER_CUBIC
        #frame = imutils.rotate(frame, angle=270)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5-3 얼굴검출기인 detector를 통해 그레이 프레임에서 을 탐지.
        rects = detector(gray, 0)  # rectangles[ [(351,137) (567,351)]  ]

        # 5-4  얼굴영역(ROI)의 갯수만큼 for문 돈다: 인식에서는 얼굴은 한개니까 한바퀴 돈다
        for rect in rects:
            # 1) pridictor
            shape = predictor(gray, rect)  # 얼굴영역(rect)의 facial landmarks (shape-> (x,y)의 list)를 정하고
            shape = face_utils.shape_to_np(shape)  # 그걸 NumPy array(ndarray)로 변환

            # 2) 배열 슬라이싱으로 좌/우 눈의 (x,y)좌표 추출
            (leftEye, rightEye) = (shape[lStart:lEnd], shape[rStart:rEnd])
            (leftEyebrow, rightEyebrow) = (shape[lBStart:lBEnd], shape[rBStart:rBEnd])

            leftEyeWidth = dist.euclidean(leftEye[0], leftEye[3])
            leftEyeHeight = dist.euclidean(leftEye[1], leftEye[4])
            rightEyeWidth = dist.euclidean(rightEye[0], rightEye[3])
            rightEyeHeight = dist.euclidean(rightEye[1], rightEye[4])

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # 관심영역 설정: 얼굴 전체영역
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 관심영역 설정: 왼쪽 눈 ROI
            (x, y, w, h) = (leftEye[0][0], leftEye[1][1], int(leftEyeWidth), int(leftEyeHeight))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # 관심영역 설정: 오른 눈 ROI
            (x, y, w, h) = (rightEye[0][0], rightEye[1][1], int(rightEyeWidth), int(rightEyeHeight))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # 관심영역 설정: 왼쪽 눈 홍채 ROI
            (x, y, w, h) = (
            leftEye[1][0], leftEye[1][1], int(leftEye[2][0] - leftEye[1][0]), int(leftEye[5][1] - leftEye[1][1]))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # 관심영역 설정: 딥러닝을 위한 ROI
            (x, y, w, h) = (rightEyebrow[0][0], leftEyebrow[2][1] - 5, int(leftEyebrow[4][0] - rightEyebrow[0][0]),
                            int(rightEye[4][1] + 10 - leftEyebrow[2][1]))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 180, 220), 1)

            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # image crop
            (start_x, start_y, end_x, end_y) = (
            leftEye[0][0] - 20, leftEye[2][1] - 20, leftEye[3][0] + 20, leftEye[5][1] + 20)
            cropped_left = frame[start_y:end_y, start_x:end_x]

            (start_x, start_y, end_x, end_y) = (
            rightEye[0][0] - 20, rightEye[2][1] - 20, rightEye[3][0] + 20, rightEye[5][1] + 20)
            cropped_right = frame[start_y:end_y, start_x:end_x]

            # image blur
            blur_left = variance_of_laplacian(cropped_left)
            blur_right = variance_of_laplacian(cropped_right)

            # cv2.putText(frame, "{}: {:.2f}".format('blur:', blur), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # cv2.imshow("Image", cropped)
            # key = cv2.waitKey(0)
            data.append((round(blur_left), round(blur_right)))

            blur = (blur_left+blur_right)/2
            '''
            # image blur detection
            if blur > 0:
                # image save
                if ear > 0.3:
                    cv2.imwrite(
                        '/home/hongbog/gen_data/right/' + str(num_name) + '/30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/home/hongbog/gen_data/left/' + str(num_name) + '/30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_30 = cnt_30 + 1
                elif ear > 0.25:
                    cv2.imwrite(
                        '/home/hongbog/gen_data/right/' + str(num_name) + '/25-30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/home/hongbog/gen_data/left/' + str(num_name) + '/25-30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_25 = cnt_25 + 1
                elif ear > 0.2:
                    cv2.imwrite(
                        '/home/hongbog/gen_data/right/' + str(num_name) + '/20-25/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/home/hongbog/gen_data/left/' + str(num_name) + '/20-25/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_20 = cnt_20 + 1
                else:
                    cv2.imwrite(
                        '/home/hongbog/gen_data/right/' + str(num_name) + '/0-20/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/home/hongbog/gen_data/left/' + str(num_name) + '/0-20/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_00 = cnt_00 + 1
            '''
            #num_name = num_name + 6
            # image blur detection
            if blur > 30:
                # image save
                if ear > 0.3:
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_30 = cnt_30 + 1
                elif ear > 0.25:
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/25-30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/25-30/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_25 = cnt_25 + 1
                elif ear > 0.2:
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/20-25/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/20-25/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_20 = cnt_20 + 1
                else:
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/right/' + str(num_name) + '/0-20/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_left)
                    cv2.imwrite(
                        '/media/sf_ShareFolder/gen_data/left/' + str(num_name) + '/0-20/' + str(frame_cnt).zfill(5) + '_' + str(
                            int(ear * 100)) + '_' + str(int(blur)) + '.png', cropped_right)
                    frame_cnt = frame_cnt + 1
                    cnt_00 = cnt_00 + 1

            # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord("q"):
            break

    print(name, ' :', cnt_30, cnt_25, cnt_20, cnt_00)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

    with open('/media/sf_ShareFolder/'+str(num_name)+'test.txt', 'w') as f:
        for d in data:
            f.write(str(d[0]) + ',' + str(d[1]) + '\n')