
import cv2


class DataWrite:

    def __init__(self, cropped_all, cropped_left, cropped_right, name, frame_cnt, blur, ear, cnt_30, cnt_25, cnt_20, cnt_00):
        self.cropped_all = cropped_all
        self.cropped_left = cropped_left
        self.cropped_right = cropped_right
        self.name = name
        self.frame_cnt = frame_cnt
        self.blur = blur
        self.ear = ear
        self.cnt_30 = cnt_30
        self.cnt_25 = cnt_25
        self.cnt_20 = cnt_20
        self.cnt_00 = cnt_00    # d

    def imgwrite(self, dirRoot):
        if self.blur > 5:   #10
            # image save
            if self.ear > 0.3:
                cv2.imwrite(dirRoot + '/all/' + str(self.name) + '/30/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_all)
                cv2.imwrite(dirRoot + '/left/' + str(self.name) + '/30/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_left)
                cv2.imwrite(dirRoot + '/right/' + str(self.name) + '/30/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_right)
                self.frame_cnt = self.frame_cnt + 1
                self.cnt_30 = self.cnt_30 + 1
            elif self.ear > 0.25:
                cv2.imwrite(dirRoot + '/all/' + str(self.name) + '/25-30/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_all)
                cv2.imwrite(dirRoot + '/left/' + str(self.name) + '/25-30/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_left)
                cv2.imwrite(dirRoot + '/right/' + str(self.name) + '/25-30/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_right)
                self.frame_cnt = self.frame_cnt + 1
                self.cnt_25 = self.cnt_25 + 1
            elif self.ear > 0.2:
                cv2.imwrite(dirRoot + '/all/' + str(self.name) + '/20-25/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_all)
                cv2.imwrite(dirRoot + '/left/' + str(self.name) + '/20-25/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_left)
                cv2.imwrite(dirRoot + '/right/' + str(self.name) + '/20-25/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_right)
                self.frame_cnt = self.frame_cnt + 1
                self.cnt_20 = self.cnt_20 + 1
            else:
                cv2.imwrite(dirRoot + '/all/' + str(self.name) + '/0-20/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_all)
                cv2.imwrite(dirRoot + '/left/' + str(self.name) + '/0-20/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_left)
                cv2.imwrite(dirRoot + '/right/' + str(self.name) + '/0-20/' + str(self.frame_cnt).zfill(5) + '_' + str(int(self.ear * 100)) + '_' + str(int(self.blur)) + '.png', self.cropped_right)
                self.frame_cnt = self.frame_cnt + 1
                self.cnt_00 = self.cnt_00 + 1

        return self.frame_cnt, self.cnt_00, self.cnt_20, self.cnt_25, self.cnt_30
