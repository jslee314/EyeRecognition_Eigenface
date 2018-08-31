'''
brightness
1. detect blur
2. transformation(translation, rotation, scale)
3.

'''
import cv2


class ImageProcessing:

    @classmethod
    def crop_eye(cls, image, leftEye, rightEye):
        ''' 수image crop 함수'''
        ''' left eye image crop '''
        start_x = leftEye[0][0]
        end_x = leftEye[3][0]

        start_y = leftEye[1][1] if leftEye[1][1] < leftEye[2][1] else leftEye[2][1]
        end_y = leftEye[4][1] if leftEye[4][1] > leftEye[5][1] else leftEye[5][1]

        h = end_y - start_y
        w = end_x - start_x

        x_scope1 = (3*h - w)/2 + 3*h*0.4
        x_scope2 = (3*h - w)/2 + 3*h*0.7
        y_scope = h * 0.7

        points = [int(start_x - x_scope1), int(start_y - y_scope), int(end_x + x_scope2), int(end_y + y_scope)] # [start_x, start_y, end_x, end_y]
        all_eye_points = [0, 0, int(end_x + x_scope2), int(end_y + y_scope)]

        # points = [point if point > 0 else 0 for point in points]           # 0 보다 작은값은 0으로
        cropped_left = image[points[1]:points[3], points[0]:points[2]]
        lw = points[2] - points[0]
        lh = points[3] - points[1]

        ''' right eye image crop '''
        start_x = rightEye[0][0]
        end_x = rightEye[3][0]

        start_y = rightEye[1][1] if rightEye[1][1] < rightEye[2][1] else rightEye[2][1]
        end_y = rightEye[4][1] if rightEye[4][1] > rightEye[5][1] else rightEye[5][1]

        h = end_y - start_y
        w = end_x - start_x

        x_scope1 = (3*h - w)/2 + 3*h*0.4
        x_scope2 = (3*h - w)/2 + 3*h*0.7
        y_scope = h * 0.7

        points = [int(start_x - x_scope2), int(start_y - y_scope), int(end_x + x_scope1), int(end_y + y_scope)] # [start_x, start_y, end_x, end_y]

        all_eye_points[0] = int(start_x - x_scope2)
        all_eye_points[1] = int(start_y - y_scope)

        #points = [point if point > 0 else 0 for point in points]           # 0 보다 작은값은 0으로
        cropped_right = image[points[1]:points[3], points[0]:points[2]]

        rw = points[2] - points[0]
        rh = points[3] - points[1]

        ''' all eye image crop '''
        cropped_all = image[ all_eye_points[1]:all_eye_points[3], all_eye_points[0]:all_eye_points[2]]
        all_w = all_eye_points[2] - all_eye_points[0]
        all_h = all_eye_points[3] - all_eye_points[1]

        return cropped_all, cropped_left, cropped_right, all_w, all_h, lw, lh, rw, rh

    # 1-1. detect blur
    @classmethod
    def variance_of_laplacian(cls, image):
        # image blur detection
        # return the focus measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    # 1-2. detect blur


    # 2. transformation(translation, rotation, scale)
    def transformation(self):
        pass



    # 3.

