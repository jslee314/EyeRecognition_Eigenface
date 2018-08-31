# Import necessary packages
from __future__ import print_function
import os
import sys
import cv2
import numpy as np
from PIL import Image

''' 
    	Allocate space for all images in one data matrix.
      The size of the data matrix is     ( w  * h  * 3, numImages )
     where,
     w = width of an image in the dataset.
     h = height of an image in the dataset.
      3 is for the 3 color channels.
     '''
# Create data matrix from a list of images
def createDataMatrix(images):
    print("Creating data matrix", end=" ... ")
    numImages = len(images)

    sz = images[0].shape
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
    print(data.shape, numImages, images[0].shape)

    for i in range(0, numImages):
        sz = images[i].shape
        if sz[0] == 160:
            if sz[1] == 400:
                image = images[i].flatten()
                data[i, :] = image

    print("Create Data Matrix DONE")
    return data


# Read images from the directory
def readImages(path):
    print("Reading images from " + path, end="...")
    # Create array of array of images.
    images = []

    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:
            # Add to array of images
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)
            if im is None:
                print("image:{} not read properly".format(imagePath))
            else:
                # Convert image to floating point
                im = np.float32(im) / 255.0
                # Add image to list
                images.append(im)
                # Flip image
                imFlip = cv2.flip(im, 1);
                # Append flipped image
                images.append(imFlip)

    numImages = len(images) / 2
    # Exit if no image found
    if numImages == 0:
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images


# Add the weighted eigen faces to the mean face
def createNewFace(*args):
    # Start with the mean image
    output = averageFace

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE / 2
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size
    output = cv2.resize(output, (0, 0), fx=2, fy=2)
    cv2.imshow("Result", output)


def resetSliderValues(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE / 2));
    createNewFace()


if __name__ == '__main__':

    # Number of EigenFaces
    NUM_EIGEN_FACES = 5

    # Maximum weight
    MAX_SLIDER_VALUE = 255

    # todo: 1) Obtain a facial image dataset
    #dirName = "images"
    dirName = "/media/sf_ShareFolder/Eigenface/imageseye/6"

    # Read images
    images = readImages(dirName)

    # todo: 2) Align and resize images (데이터 세트의 모든 이미지는 동일한 크기 여야함)
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

    # todo: 3) Create data matrix for PCA. (모든 이미지를 포함하는 데이터 매트릭스를 행 벡터로 만)
    data = createDataMatrix(images)

    # todo: 4) Calculate Mean Vector (opencv는 자동으로 평균을 산출하기 때문에 계산할 필요x)

    # todo: 5) Calculate 'Principal Components' (공분산 행렬의 고유 벡터를 찾아 계산)
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print(mean, eigenVectors)

    # todo: 6) Reshape Eigenvectors to obtain 'EigenFaces'
    averageFace = mean.reshape(sz)

    eigenFaces = [];
    i = 0
    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)
        i=i+1
        with open('/media/sf_ShareFolder/Eigenface/eigenFace'+str(i)+'.txt', 'w') as f:
            for d in eigenFace:
                f.write( str(d[0]) + ' ,' + str(d[1]) + ' ,' + str(d[2]) + ' ,' + str(d[3]) + ' ,' + str(d[4]) + ' ,' + str(
                        d[5]) + ' ,' + str(d[6]) + ' ,' + str(d[7]) + ' ,' + str(d[8]) + ' ,' + str(d[9]) + '\n')

        '''
        # Create window for displaying eigenFace
        cv2.namedWindow("Result" + str(eigenVector), cv2.WINDOW_AUTOSIZE)
        output = np.add(output, eigenFaces[i])
        output = cv2.resize(output, (0, 0), fx=1, fy=1)
        cv2.imshow("Result" + str(eigenVector), output)
        i = i + 1
        '''
    with open('/media/sf_ShareFolder/Eigenface/eigenFace.txt', 'w') as f:
        for d in eigenFaces:
            f.write(str(d[0]) + ' ,' + str(d[1]) + ' ,' + str(d[2]) + ' ,' + str(d[3]) + ' ,' + str(d[4]) + ' ,' + str(d[5]) + ' ,' + str(d[6]) + ' ,' + str(d[7]) + ' ,' + str(d[8]) + ' ,' + str(d[9]) + '\n')

    with open('/media/sf_ShareFolder/Eigenface/eigenVector.txt', 'w') as f:
        for d in eigenVectors:
            f.write(str(d[0])+' ,' + str(d[1])+' ,' + str(d[2]) + ' ,' + str(d[3]) + ' ,' + str(d[4]) + ' ,' + str(d[5]) + ' ,' + str(d[6]) + ' ,' + str(d[7]) + ' ,' + str(d[8]) + ' ,' + str(d[9])+ '\n')

    #-----------------------------------testing




    # Create window for displaying Mean Face
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
    # Display result at 2x size
    output = cv2.resize(averageFace, (0, 0), fx=2, fy=2)
    cv2.imshow("Result", output)


    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    sliderValues = []

    # Create Trackbars
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(MAX_SLIDER_VALUE / 2)
        cv2.createTrackbar("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE / 2), MAX_SLIDER_VALUE, createNewFace)

    # You can reset the sliders by clicking on the mean image.
    cv2.setMouseCallback("Result", resetSliderValues);

    print('''Usage:
	Change the weights using the sliders
	Click on the result window to reset sliders
	Hit ESC to terminate program.''')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
