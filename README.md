# Eigenface을 통해 눈모양 인식 구현

### Eigenface(주성분 분석(PCA)을 통해 얻은 주성분 벡터들)을 통해 실제 얼굴이미지을 적용하여 얼굴인식을 테스트하고, 이 알고리즘으로 눈모양 인식을 적용



## STEP 1) Obtain a facial image dataset
    #dirName = "images"
    dirName = "/media/sf_ShareFolder/Eigenface/imageseye/6"

    # Read images
    images = readImages(dirName)


## STEP 2) Align and resize images (데이터 세트의 모든 이미지는 동일한 크기 여야함)
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
    
 
## STEP 3) Create data matrix for PCA. (모든 이미지를 포함하는 데이터 매트릭스를 행 벡터로 만)
    data = createDataMatrix(images)


## STEP 4) Calculate Mean Vector (opencv는 자동으로 평균을 산출하기 때문에 계산할 필요x)


## STEP 5) Calculate 'Principal Components' (공분산 행렬의 고유 벡터를 찾아 계산)
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print(mean, eigenVectors)


## STEP 6) Reshape Eigenvectors to obtain 'EigenFaces'

    averageFace = mean.reshape(sz)

