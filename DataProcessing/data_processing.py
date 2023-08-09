import os
import json
from functools import reduce

import numpy as np
import cv2
from tqdm import tqdm

from Lib import UtilLib, DragonLib
from Common import const_var_dataprocessing


class DatasetMaker:
    def __init__(self, json_dir, lsgan_dir_train, lsgan_dir_test, cyclegan_dir_train_a, cyclegan_dir_test_a, singan_dir_train):
        """
        * 데이터셋 만들기
        :param json_dir: JSON 파일 해당 디렉터리
        :param lsgan_dir_train: 데이터셋 저장될 디렉터리 (LSGAN train)
        :param lsgan_dir_test: 데이터셋 저장될 디렉터리 (LSGAN test)
        :param cyclegan_dir_train_a: 데이터셋 저장될 디렉터리 (CycleGAN train a)
        :param cyclegan_dir_test_a: 데이터셋 저장될 디렉터리 (CycleGAN test a)
        :param singan_dir_train: 데이터셋 저장될 디렉터리 (SinGAN train)
        """

        # JSON 파일 해당 디렉터리
        self.json_dir = json_dir
        # 데이터셋 저장될 디렉터리 (LSGAN train)
        self.lsgan_dir_train = lsgan_dir_train
        # 데이터셋 저장될 디렉터리 (LSGAN test)
        self.lsgan_dir_test = lsgan_dir_test
        # 데이터셋 저장될 디렉터리 (CycleGAN train a)
        self.cyclegan_dir_train_a = cyclegan_dir_train_a
        # 데이터셋 저장될 디렉터리 (CycleGAN test a)
        self.cyclegan_dir_test_a = cyclegan_dir_test_a
        # 데이터셋 저장될 디렉터리 (SinGAN train)
        self.singan_dir_train = singan_dir_train

        # 변환된 데이터 담을 리스트
        self.processed_dict_list = []

    def running(self):
        """
        * 데이터셋 만들기 진행
        :return: 데이터셋 만들어서 저장됨
        """

        # JSON 파일에서 정보 읽어 이미지 파일 만들기
        self._pre_processing()
        # 만들어진 이미지 파일을 데이터셋 형식에 맞게 변환
        self._post_processing()
        # 이미지 데이터셋 각각 저장
        self._save_dataset()

    def _pre_processing(self):
        """
        * JSON 파일에서 정보 읽어 이미지 파일 만들기
        :return: self.processed_dict_list 만들어짐
        """

        # JSON 파일 순회
        for json_file_name in tqdm(iterable=os.listdir(self.json_dir), desc='pre_processing'):
            json_path = UtilLib.getNewPath(path=self.json_dir, add=json_file_name)

            with open(json_path, encoding='UTF-8') as json_file:
                json_data = json.load(json_file)

                # 최소, 최대 점
                x_min, y_min = (np.array(json_data[const_var_dataprocessing.KEY_PRE_BBOX][const_var_dataprocessing.KEY_PRE_BBOX_MIN]) * const_var_dataprocessing.METER2PIXEL).astype(np.uint16)
                x_max, y_max = (np.array(json_data[const_var_dataprocessing.KEY_PRE_BBOX][const_var_dataprocessing.KEY_PRE_BBOX_MAX]) * const_var_dataprocessing.METER2PIXEL).astype(np.uint16)

                # 모든 점
                points = (np.array(json_data[const_var_dataprocessing.KEY_PRE_VERTS]) * const_var_dataprocessing.METER2PIXEL).astype(np.int32)
                points[:, 0] = points[:, 0] - x_min + const_var_dataprocessing.BORDER_PAD
                points[:, 1] = points[:, 1] - y_min + const_var_dataprocessing.BORDER_PAD

                # 그리기
                image = np.zeros(shape=(y_max - y_min + const_var_dataprocessing.BORDER_PAD * 2, x_max - x_min + const_var_dataprocessing.BORDER_PAD * 2), dtype=np.uint8)
                cv2.drawContours(image=image, contours=[points, ], contourIdx=0, color=255, thickness=const_var_dataprocessing.BORDER_PAD * 2)

                # 데이터 담기
                self.processed_dict_list.append({
                    const_var_dataprocessing.KEY_POST_IMAGE: image,
                    const_var_dataprocessing.KEY_POST_ID: json_data[const_var_dataprocessing.KEY_PRE_ID],
                    const_var_dataprocessing.KEY_POST_ROOM_NUM: json_data[const_var_dataprocessing.KEY_PRE_ROOM_NUM]
                })

    def _post_processing(self):
        """
        * 검은 배경 위에 도면 그림 붙여 넣어 동일한 사이즈를 갖는 이미지 데이터셋 만들기
        :return: self.processed_dict_list 수정됨
        """

        # 너비 혹은 길이가 MAX_LENGTH 보다 큰 이미지들은 버리기
        self.processed_dict_list = [processed_dict for processed_dict in self.processed_dict_list if max(*processed_dict[const_var_dataprocessing.KEY_POST_IMAGE].shape) < const_var_dataprocessing.MAX_LENGTH]
        # MAX_LENGTH 보다 작은 범위 안에서 최대 값 max_length 구하기
        max_length = reduce(lambda current_max_length, processed_dict: max(current_max_length, *processed_dict[const_var_dataprocessing.KEY_POST_IMAGE].shape),
                            self.processed_dict_list,
                            const_var_dataprocessing.INITIAL_MAX_LENGTH)

        # 동일한 사이즈를 갖는 이미지 만들기
        for processed_dict in tqdm(iterable=self.processed_dict_list, desc='post_processing'):
            # 검은 배경 위에 그림 붙여 넣기
            background = np.zeros(shape=(max_length, max_length), dtype=np.uint8)
            y_start = (max_length - processed_dict[const_var_dataprocessing.KEY_POST_IMAGE].shape[0]) // 2
            y_end = y_start + processed_dict[const_var_dataprocessing.KEY_POST_IMAGE].shape[0]
            x_start = (max_length - processed_dict[const_var_dataprocessing.KEY_POST_IMAGE].shape[1]) // 2
            x_end = x_start + processed_dict[const_var_dataprocessing.KEY_POST_IMAGE].shape[1]
            background[y_start:y_end, x_start:x_end] = processed_dict[const_var_dataprocessing.KEY_POST_IMAGE]
            processed_dict[const_var_dataprocessing.KEY_POST_IMAGE] = background

    def _save_dataset(self):
        """
        * 이미지 데이터셋 각각 저장하기
        :return: 이미지 데이터셋 각각 해당 경로에 저장됨
        """

        # 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(self.lsgan_dir_train)
        DragonLib.make_parent_dir_if_not_exits(self.lsgan_dir_test)
        DragonLib.make_parent_dir_if_not_exits(self.cyclegan_dir_train_a)
        DragonLib.make_parent_dir_if_not_exits(self.cyclegan_dir_test_a)
        DragonLib.make_parent_dir_if_not_exits(self.singan_dir_train)

        # 학습 / 테스트 데이터 70%, 30% 로 나누기 위한 값
        train_data_num = int(len(self.processed_dict_list) * 0.7)

        # 데이터셋 저장 진행
        for num, processed_dict in enumerate(iterable=tqdm(iterable=self.processed_dict_list,
                                                           desc='save_processing'),
                                             start=1):
            image_file_name = const_var_dataprocessing.IMAGE_FILE_NAME.format(processed_dict[const_var_dataprocessing.KEY_POST_ID], processed_dict[const_var_dataprocessing.KEY_POST_ROOM_NUM])

            # LSGAN train 데이터 셋 저장
            # CycleGAN train a 데이터 셋 저장
            # SinGAN train 데이터 셋 저장
            if num <= train_data_num:
                cv2.imwrite(filename=self.lsgan_dir_train + image_file_name, img=processed_dict[const_var_dataprocessing.KEY_POST_IMAGE])
                cv2.imwrite(filename=self.cyclegan_dir_train_a + image_file_name, img=processed_dict[const_var_dataprocessing.KEY_POST_IMAGE])
                # SinGAN train 데이터는 NUM_SINGAN_TRAIN (1) 개 까지만 담기
                if num <= const_var_dataprocessing.NUM_SINGAN_TRAIN:
                    cv2.imwrite(filename=self.singan_dir_train + image_file_name, img=processed_dict[const_var_dataprocessing.KEY_POST_IMAGE])
            # LSGAN test 데이터 셋 저장
            # CycleGAN test a 데이터 셋 저장
            else:
                # PROPER_MAX_NUM_TEST (200) 개 까지만 담기
                if num - train_data_num <= const_var_dataprocessing.PROPER_MAX_NUM_TEST:
                    cv2.imwrite(filename=self.lsgan_dir_test + image_file_name, img=processed_dict[const_var_dataprocessing.KEY_POST_IMAGE])
                    cv2.imwrite(filename=self.cyclegan_dir_test_a + image_file_name, img=processed_dict[const_var_dataprocessing.KEY_POST_IMAGE])
