'''
좌/우 눈과 눈썹의 landmark를 구한 후 눈인식을 위한특징을 추출하는 함수
1. 좌/우 눈의 미간 의거리.
2. 좌/우 눈의 동그란정도 blink detection 에 사용된다.
3. 좌/우 눈의 눈과 눈썹 사이 길이비교.
4. 좌/우 눈의 양끝점 의각 도 :눈꼬리가 올라간 눈 인지, 처진 눈 인지)
5. 좌/우 눈의 망막 의크기 와홍채 의크 기 비율. :검은눈동자 가얼마 나큰지
6. 두눈의 양끝점을 이은 선을 중심으로 동공의 중심 위치가 위쪽 또는 아래쪽에 포함되는지 비교.
7.........
'''
from scipy.spatial import distance as dist
import math
import cv2
# from matplotlib import pyplot as plt


class FeaturePoint:
    def __init__(self, eye_left, eye_right, eyebrow_left, eyebrow_right):
        self.eye_left = eye_left
        self.eye_right = eye_right
        self.eyebrow_left = eyebrow_left
        self.eyebrow_right = eyebrow_right

    # 1. EBR: 좌/우 눈의 미간 의거리. EBR
    def eye_between_ratio(self):

        left_width = dist.euclidean(self.eye_left[0], self.eye_left[3])
        right_width = dist.euclidean(self.eye_right[0], self.eye_right[3])
        btw_width = dist.euclidean((self.eye_left[0]), self.eye_right[3])

        ebr = btw_width / (left_width + right_width + btw_width)

        return ebr

    # 2. EAR: 좌/우 눈의 동그란정도, blink detection 에 사용된다.
    def eye_aspect_ratio(self):
        a = dist.euclidean(self.eye_left[1], self.eye_left[5])
        b = dist.euclidean(self.eye_left[2], self.eye_left[4])
        c = dist.euclidean(self.eye_left[0], self.eye_left[3])
        left_ear = (a + b) / (2.0 * c)

        a = dist.euclidean(self.eye_right[1], self.eye_right[5])
        b = dist.euclidean(self.eye_right[2], self.eye_right[4])
        c = dist.euclidean(self.eye_right[0], self.eye_right[3])
        right_ear = (a + b) / (2.0 * c)

        # mean of ear
        ear = (left_ear + right_ear) / 2.0

        return left_ear, right_ear, ear

    # 3. EEBR: 좌/우 눈의 눈과 눈썹 사이 길이비교.
    def eye_eyebrow_ratio(self):
        for index in range(5):
            left_eebr = dist.euclidean(self.eyebrow_left[index], self.eye_left[1])
            right_eebr = dist.euclidean(self.eyebrow_right[index], self.eye_right[1])

        return left_eebr, right_eebr

    # 4. ELA: 좌/우 눈의 양끝점의 각도 :눈꼬리가 올라간 눈 인지, 처진 눈 인지)
    def eye_lines_angle(self):
        # tan(angle) = a/b
        x = self.eye_left[3][0] - self.eye_left[0][0]
        y = self.eye_left[3][1] - self.eye_left[0][1]
        left_angle = math.atan2(y, x)

        x = self.eye_right[3][0] - self.eye_right[0][0]
        y = self.eye_right[3][1] - self.eye_right[0][1]
        right_angle = math.atan2(y, x)

        return left_angle, right_angle


class FeatureImage:
    def __init__(self, image, eye_left, eye_right, eyebrow_left, eyebrow_right):
            self.image = image
            self.eye_left = eye_left
            self.eye_right = eye_right
            self.eyebrow_left = eyebrow_left
            self.eyebrow_right = eyebrow_right

    # 5. EIR: 좌/우 눈의 망막의 크기와 홍채의 크기 비율. :검은눈동자 가얼마 나큰지
    def eye_iris_ratio(self):
        cropped_left, cropped_right = self._eye_crop(self.image, self.eye_left, self.eye_right)
        cropped_left = cv2.cvtColor(cropped_left, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(cropped_left, 80, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(cropped_left, 80, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(cropped_left, 50, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(cropped_left, 50, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(cropped_left, 50, 255, cv2.THRESH_TOZERO_INV)

        titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        images = [cropped_left, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(6):
            cv2.imwrite('/media/sf_ShareFolder/' + str(titles[i]) + '.png', images[i])
            # plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
            # plt.title(titles[i])
            # plt.xticks([]), plt.yticks([])

        # plt.show()

        # if(thresh1 > )
        # return left_eir, right_eir

    # 6. EPL: 두눈의 양끝점을 이은 선을 중심으로 동공의 중심 위치가 위쪽 또는 아래쪽에 포함되는지 비교.
    def eye_pupil_location(self):

        return

    def _eye_crop(self):
        (start_x, start_y, end_x, end_y) = (self.leftEye[0][0] - 20, self.leftEye[2][1] - 20, self.leftEye[3][0] + 20, self.leftEye[5][1] + 20)
        cropped_left = self.image[start_y:end_y, start_x:end_x]

        (start_x, start_y, end_x, end_y) = (self.rightEye[0][0] - 20, self.rightEye[2][1] - 20, self.rightEye[3][0] + 20, self.rightEye[5][1] + 20)
        cropped_right = self.image[start_y:end_y, start_x:end_x]

        return cropped_left, cropped_right

