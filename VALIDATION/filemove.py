import numpy as np
import os
import re
import shutil

class DataSeparator:
    '''
        Train / Test 데이터 셋으로 분리하는 클래스
    '''
    def __init__(self):
        '''
            self.asis_data_path: original dataset root path
            self.tobe_data_path: new dataset root path
            D:\Oracle\ShareFolder
                    '''
        self.asis_data_path = '/media/sf_ShareFolder/SegEye/gen_data'
        self.tobe_data_path = '/media/sf_ShareFolder/SegEye/dataset'


        ''' re.compile('.*\\\\(20-25|25-30|30)\\\\.*'): 정규식을 통해 EAR 25 이상에 해당하는 path 만 추출 '''
        self.path_reg = re.compile('.*\/(20-25|25-30|30)\/.*')
        self.class_reg = re.compile('.*\/(right|left|all)\/(\d+)(\-|\_)(\d+)\/.*')


        ''' original dataset path 를 담는 변수들 '''
        self.asis_full_paths = []
        self.asis_right_paths = []
        self.asis_left_paths = []
        self.asis_all_paths = []

        '''
            new dataset path 를 담는 변수들
            train: 70%, test: 30%
        '''
        self.train_all_paths, self.test_all_paths = [], []
        self.train_left_paths, self.test_left_paths = [], []
        self.train_right_paths, self.test_right_paths = [], []

    def path_setting(self):
        '''
            새로운 데이터 셋으로 구성하기 위해 원본 이미지 파일 경로를 추출하는 함수
            :return: None
        '''
        for path, dir, filenames in os.walk(self.asis_data_path):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                if self.path_reg.search(file_path):
                    self.asis_full_paths.append(file_path)

        self.asis_full_paths = sorted(self.asis_full_paths)
        self.asis_all_paths = np.array(self.asis_full_paths[:int(len(self.asis_full_paths) / 3)])
        self.asis_left_paths = np.array(self.asis_full_paths[int(len(self.asis_full_paths) / 3):int(len(self.asis_full_paths) * 2/ 3)])
        self.asis_right_paths = np.array(self.asis_full_paths[int(len(self.asis_full_paths) * 2/ 3):])

        data_len = self.asis_left_paths.shape[0]
        random_sort = np.random.permutation(data_len)
        self.asis_left_paths = self.asis_left_paths[random_sort]
        self.train_left_paths, self.test_left_paths = self.asis_left_paths[:int(data_len * 0.7)], self.asis_left_paths[int(data_len * 0.7):]

        self.asis_right_paths = self.asis_right_paths[random_sort]
        self.train_right_paths, self.test_right_paths = self.asis_right_paths[:int(data_len * 0.7)], self.asis_right_paths[int(data_len * 0.7):]

        self.asis_all_paths = self.asis_all_paths[random_sort]
        self.train_all_paths, self.test_all_paths = self.asis_all_paths[:int(data_len * 0.7)], self.asis_all_paths[int(data_len * 0.7):]

    def file_move(self):
        self.path_setting()

        ''' train data set '''
        print('>> Train Dataset Move Start...')
        for train_path in zip(self.train_all_paths, self.train_left_paths, self.train_right_paths):
            class_num = self.class_reg.search(train_path[0]).group(2)
            sep_type = self.class_reg.search(train_path[0]).group(3)
            group_num = self.class_reg.search(train_path[0]).group(4)
            tobe_all_path = os.path.join(self.tobe_data_path, 'train', 'all', class_num)
            tobe_left_path = os.path.join(self.tobe_data_path, 'train', 'left', class_num)
            tobe_right_path = os.path.join(self.tobe_data_path, 'train', 'right', class_num)

            os.makedirs(tobe_all_path, exist_ok=True)
            os.makedirs(tobe_left_path, exist_ok=True)
            os.makedirs(tobe_right_path, exist_ok=True)

            shutil.move(train_path[0], os.path.join(tobe_all_path, str(group_num) + sep_type + os.path.basename(train_path[0])))
            shutil.move(train_path[1], os.path.join(tobe_left_path, str(group_num) + sep_type + os.path.basename(train_path[1])))
            shutil.move(train_path[2], os.path.join(tobe_right_path, str(group_num) + sep_type + os.path.basename(train_path[2])))

        print('>> Train Dataset Move End ...')

        ''' test data set '''
        print('>> Test Dataset Move Start...')
        for test_path in zip(self.test_all_paths, self.test_left_paths, self.test_right_paths ):
            class_num = self.class_reg.search(test_path[0]).group(2)
            #sep_type = self.class_reg.search(train_path[0]).group(3)
            sep_type = self.class_reg.search(test_path[0]).group(3)
            group_num = self.class_reg.search(test_path[0]).group(4)

            tobe_left_path = os.path.join(self.tobe_data_path, 'test', 'left', class_num)
            tobe_right_path = os.path.join(self.tobe_data_path, 'test', 'right', class_num)
            tobe_all_path = os.path.join(self.tobe_data_path, 'test', 'all', class_num)

            os.makedirs(tobe_left_path, exist_ok=True)
            os.makedirs(tobe_right_path, exist_ok=True)
            os.makedirs(tobe_all_path, exist_ok=True)

            shutil.move(test_path[0], os.path.join(tobe_all_path, str(group_num) + sep_type + os.path.basename(test_path[0])))
            shutil.move(test_path[1], os.path.join(tobe_left_path, str(group_num) + sep_type + os.path.basename(test_path[1])))
            shutil.move(test_path[2], os.path.join(tobe_right_path, str(group_num) + sep_type + os.path.basename(test_path[2])))

        print('>> Test Dataset Move End ...')

ds = DataSeparator()
ds.file_move()