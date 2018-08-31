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
from VALIDATION.filemove import DataSeparator


# 1. constant
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

# 2. dlib의 얼굴 검출기와 얼굴 표식 탐지기를 초기화 (dlib)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/hongbog/Project/Source/shape_predictor_68_face_landmarks.dat")
# dlib의  get_frontal_face_detector를 이용해서 탐지
fa = FaceAligner(predictor, desiredFaceWidth=720)

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

    dirRoot = '/media/sf_ShareFolder/SegEye/gen_data'

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
        #frame = imutils.resize(frame, width=360, height=640)  # inter=cv2.INTER_CUBIC
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
            shape = predictor(gray, rect)               # 얼굴영역(rect)의 facial landmarks (shape-> (x,y)의 list)를 정하고
            shape = face_utils.shape_to_np(shape)       # 그걸 NumPy array(ndarray)로 변환


            # 2) 배열 슬라이싱으로 좌/우 눈의 (x,y)좌표 추출
            (leftEye, rightEye) = (shape[lStart:lEnd], shape[rStart:rEnd])
            (leftEyebrow, rightEyebrow) = (shape[lBStart:lBEnd], shape[rBStart:rBEnd])

            # 3) Point based feature / image based feature의 인스턴스 선언
            p = FeaturePoint(leftEye, rightEye, leftEyebrow, rightEyebrow)
            i = FeatureImage(faceAligned, leftEye, rightEye, leftEyebrow, rightEyebrow)

            ''' image feature extraction '''
            # 1. EBR: 좌/우 눈의 미간 의거리. EBR
            ebr = p.eye_between_ratio()

            # 2. EAR: 좌/우 눈의 동그란정도, blink detection 에 사용된다.
            leftEAR, rightEAR, ear = p.eye_aspect_ratio()

            # 3. EEBR: 좌/우 눈의 눈과 눈썹 사이 길이비교.
            left_eebr, right_eebr = p.eye_eyebrow_ratio()

            # 4. ELA: 좌/우 눈의 양끝점의 각도 :눈꼬리가 올라간 눈 인지, 처진 눈 인지)
            left_angle, right_angle = p.eye_lines_angle()

            # 5. EIR: 좌/우 눈의 망막의 크기와 홍채의 크기 비율. :검은눈동자 가얼마 나큰지
            # p.eye_iris_ratio()

            '''  관심영역 설정  '''
            # 얼굴 전체영역
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #  눈 볼록껍질 영역
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(faceAligned, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(faceAligned, [rightEyeHull], -1, (0, 255, 0), 1)

            if (leftEAR > 0.10) and (rightEAR > 0.10):     # 0.20
                # Featuredata append
                featuredata.append((ebr, leftEAR, rightEAR, left_eebr, right_eebr, left_angle, right_angle))

                ''' image processing '''
                # image crop - eye only
                cropped_all, cropped_left, cropped_right, all_w, all_h, lw, lh, rw, rh = ip.crop_eye(faceAligned, leftEye, rightEye)

                # 1. blur
                blur_left = ip.variance_of_laplacian(cropped_left)
                blur_right = ip.variance_of_laplacian(cropped_right)
                blur = (blur_left + blur_right) / 2
                if (lw > 100) and (rw > 100):
                    #minw = min(minw, lw, rw)
                    #maxw = max(maxw, lw, rw)
                    #minh = min(minh, lh, rh)
                    #maxh = max(maxh, lh, rh)

                    minw = min(minw, all_w)
                    maxw = max(maxw, all_w)
                    minh = min(minh, all_h)
                    maxh = max(maxh, all_h)

                    # 2. illumination
                    ''' image write '''
                    # Infodata append
                    infodata.append((round(blur_left), round(blur_right)))

                    # DataWrite
                    writer = DataWrite(cropped_all, cropped_left, cropped_right, name, frame_cnt, blur, ear, cnt_30, cnt_25, cnt_20, cnt_00)
                    frame_cnt, cnt_00, cnt_20, cnt_25, cnt_30 = writer.imgwrite(dirRoot)

        # show the frame
        #cv2.imshow("Frame", faceAligned)
        key = cv2.waitKey(1) & 0xFF

    print(name, ':', cnt_30, cnt_25, cnt_20, cnt_00)

    ds = DataSeparator()
    ds.file_move()

    print('size: '+'minw:', minw, 'minh:', minh, 'maxw:', maxw, 'maxh:', maxh)

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

