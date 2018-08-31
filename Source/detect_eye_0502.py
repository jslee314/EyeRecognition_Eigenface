
from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import glob

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

from VALIDATION.detect_eye_feature import FeaturePoint
from VALIDATION.detect_eye_feature import FeatureImage
from VALIDATION.image_processing import ImageProcessing as ip
from VALIDATION.datawrite import DataWrite
import VALIDATION.dataload as dl
# from VALIDATION.filemove import DataSeparator


# 1. constant
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

# 2. dlib의 얼굴 검출기와 얼굴 표식 탐지기를 초기화 (dlib)
detector = dlib.get_frontal_face_detector()
predictor_2 = dlib.shape_predictor("/home/hongbog/Project/Source/shape_predictor_5_face_landmarks.dat")
# dlib의  get_frontal_face_detector를 이용해서 탐지
fa_2 = FaceAligner(predictor_2, desiredFaceWidth=720)

# 3. 좌/우 눈의 indexing 값 가져오기 (array slice index values)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lBStart, lBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rBStart, rBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

# 4. mp4 파일 가져오기
list_name = glob.glob("/media/sf_ShareFolder/SegEye/data/mp4/run/*.mp4")
maxw = 0
maxh = 0

minw = 999999
minh = 999999

'''
dicgamm = dict()
for val in range(256):
    dicgamm[val] = round(val ** (1.0 / 1.3))  # dicgamm[val] = round(math.sqrt(val))
'''

for num_name in range(len(list_name)):

    filename = list_name[num_name]
    # 비디오스트림 시작
    vs = FileVideoStream(filename).start()
    fileStream = True
    time.sleep(1.0)

    name = filename[-8:-4]

    dirRoot = '/media/sf_ShareFolder/SegEye/gen_data111'

    dl.makedir(name, dirRoot)

    frame_cnt = 1

    # output data
    data = []
    infodata = []
    featuredata = []


    cnt_30 = 0
    cnt_25 = 0
    cnt_20 = 0
    cnt_00 = 0


    # 5. 비디오 스트림 반복 루프------------------------------------------------
    while True:
        # 5-1 비디오에 프레임이 없으면 루프 탈출.
        if fileStream and not vs.more():
            break

        # 5-2 이미지 전처리: 스트림에서 프레임을 읽은 다음, 크기를 조정하고, 회색 음영으로 변환.
        frame = vs.read()
        #frame = imutils.resize(frame, width=1080, height=1920)  # inter=cv2.INTER_CUBIC
        #frame = imutils.rotate(frame, angle=90)
        frame = imutils.resize(frame, width=720, height=1280)  # inter=cv2.INTER_CUBIC

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #graygamma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", grayy)
        #key = cv2.waitKey(1) & 0xFF

        #gray = cv2.equalizeHist(grayy)
        #cv2.imshow("equalizeHist", gray)
        #key = cv2.waitKey(1) & 0xFF
        '''
        for y in range(1280):
            for x in range(720):
                graygamma[y][x] = dicgamm[gray[y][x]]

        '''
        '''
        graygamma = []
        for y in range(1280):
            temp = []
            for x in range(720):
                temp.append(dicgamm[grayy[y][x]])
            graygamma.append(temp)
        '''
        #cv2.imshow("graygamma", graygamma)
        #key = cv2.waitKey(1) & 0xFF

        # 5-3 얼굴검출기인 detector를 통해 그레이 프레임에서 을 탐지.
        rects = detector(gray, 0)  # rectangles[ [(351,137) (567,351)]  ]
        '''
        for rect in rects:
            # 0) extract the ROI of the *original* face, then align the face using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=720)
            faceAligned = fa.align(frame, gray, rect)
        '''
        faceAligned = frame
        # 5-4  얼굴영역(ROI)의 갯수만큼 for문 돈다: 인식에서는 얼굴은 한개니까 한바퀴 돈다
        for rect in rects:

            # 1) pridictor
            shape_2 = predictor_2(gray, rect)               # 얼굴영역(rect)의 facial landmarks (shape-> (x,y)의 list)를 정하고
            shape_2 = face_utils.shape_to_np(shape_2)       # 그걸 NumPy array(ndarray)로 변환

            # 2) 배열 슬라이싱으로 좌/우 눈의 (x,y)좌표 추출
            point5 = shape_2[0:5]
            (point5[2][1], point5[1][1]) = (point5[2][1] - 20, point5[1][1] - 20)
            (point5[3][1], point5[1][1]) = (point5[0][1] + 20, point5[1][1] + 20)

            (leftEye, rightEye) = (point5[0:2], point5[2:4])
            (leftEyebrow, rightEyebrow) = (point5[0:1], point5[2:3])

            # 3) Point based feature / image based feature의 인스턴스 선언
            p = FeaturePoint(leftEye, rightEye, leftEyebrow, rightEyebrow)
            i = FeatureImage(faceAligned, leftEye, rightEye, leftEyebrow, rightEyebrow)

            ''' image feature extraction '''


            # 2. EAR: 좌/우 눈의 동그란정도, blink detection 에 사용된다.
            leftEAR, rightEAR, ear = (0.3, 0.3, 0.3)

            '''  관심영역 설정  '''
            # 얼굴 전체영역
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #  눈 볼록껍질 영역
            #leftEyeHull = cv2.convexHull(leftEye)
            #rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(faceAligned, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(faceAligned, [rightEyeHull], -1, (0, 255, 0), 1)

            w = shape_2[0][0] - shape_2[1][0]
            h = 40
            (x, y, w, h) = (point5[1][0], point5[2][1], int(w), int(h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            w = shape_2[3][0] - shape_2[2][0]
            h = 40
            (x, y, w, h) = (point5[2][0], point5[2][1], int(w), int(h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)



        # show the frame
        cv2.imshow("Frame", faceAligned)
        key = cv2.waitKey(1) & 0xFF

    print(name, ':', cnt_30, cnt_25, cnt_20, cnt_00)

    #ds = DataSeparator()
    #ds.file_move()

    print('minw:', minw, 'minh:', minh, 'maxw:', maxw, 'maxh:', maxh)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

    with open(dirRoot + '/featureData' + str(name) + '.txt', 'w') as f:
        for d in featuredata:
            f.write(str(d[0]) + ' ,' + str(d[1]) + ' ,' + str(d[2]) + ' ,' + str(d[3]) + ' ,' + str(d[4]) + ' ,' + str(d[5]) + ' ,' + str(d[6]) + '\n')
'''
    with open(dirRoot + '/data' +str(name) + '.txt', 'w') as f:
        for d in data:
            f.write(str(d[0]) + ' ,' + str(d[1]) + ' ,' + str(d[2]) + ' ,' + str(d[3]) + '\n')

    with open(dirRoot + '/infodata'+str(name) + '.txt', 'w') as f:
        for d in infodata:
            f.write(str(d[0]) + ' ,' + str(d[1]) + '\n')
'''