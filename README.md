# Eigenface을 통해 눈모양 인식 구현

### Eigenface(주성분 분석(PCA)을 통해 얻은 주성분 벡터들)을 통해 실제 얼굴이미지을 적용하여 얼굴인식을 테스트하고, 이 알고리즘으로 눈모양 인식을 적용



#### STEP 1) Obtain a facial image dataset
    #dirName = "images"
    dirName = "/media/sf_ShareFolder/Eigenface/imageseye/6"

    # Read images
    images = readImages(dirName)


#### STEP 2) Align and resize images (데이터 세트의 모든 이미지는 동일한 크기 여야함)
    # images resize
    for image in images:
        # r = 200.0 / image.shape[1]
        #dim = (200, int(image.shape[0] * r))
        dim = (400, 160)
        #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        #cv2.imshow("resized", image)

    # Size of images
    sz = images[0].shape
    
 
#### STEP 3) Create data matrix for PCA. (모든 이미지를 포함하는 데이터 매트릭스를 행 벡터로 만)
    data = createDataMatrix(images)


#### STEP 4) Calculate Mean Vector (opencv는 자동으로 평균을 산출하기 때문에 계산할 필요x)


#### STEP 5) Calculate 'Principal Components' (공분산 행렬의 고유 벡터를 찾아 계산)
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print(mean, eigenVectors)


#### STEP 6) Reshape Eigenvectors to obtain 'EigenFaces'

    averageFace = mean.reshape(sz)


---------------------------------------------------------------------------------------------------------------------------
# 눈 인식을 위해 얼굴에서 눈 영역 추출

사용 라이브러리 : dlib http://dlib.net/


#### STEP 1) dlib라이브러리를 사용하여 얼굴 검출기와 얼굴 표식 탐지기를 초기화
#### STEP 2) 좌/우 눈과 눈썹의 landmark를 구한 후 눈인식을 위한특징을 추출
#### STEP 3) 좌/우 눈의 indexing 값 가져오기 (array slice index values)
#### STEP 4) mp4 파일 가져오기
#### STEP 5) 비디오 스트림 반복 (비디오에 프레임이 없으면 루프 탈출!)
#### STEP 5-1)  이미지 전처리: 스트림에서 프레임을 읽은 다음, 크기를 조정하고, 회색 음영으로 변환.
#### STEP 5-2)  얼굴검출기인 detector를 통해 그레이 프레임에서 을 탐지
#### STEP 5-3) 얼굴영역(ROI)의 갯수만큼 for문 돈다: 인식에서는 얼굴은 한개니까 한바퀴 돈다 (두명의 얼굴이 있으면 두바퀴 돈다)

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













![image](https://user-images.githubusercontent.com/40026846/115135006-1bd98480-a050-11eb-97bd-2710bdea83eb.png)

