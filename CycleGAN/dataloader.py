import os, random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from Lib import UtilLib
from Common import const_var_cyclegan


class FakeHome2RealHomeDataset(Dataset):
    def __init__(self, data_dir, mode_train_test):
        """
        * FakeHome2RealHomeDataset 데이터로더
        :param data_dir: 데이터 디렉터리
        :param mode_train_test: 학습 / 테스트 모드
        """

        # 데이터 해당 디렉터리
        self.data_dir = data_dir
        # 학습 / 테스트 모드
        self.mode_train_test = mode_train_test

        # a 데이터 디렉터리
        self.dir_a = UtilLib.getNewPath(path=self.data_dir,
                                        add='a')
        # b 데이터 디렉터리
        self.dir_b = UtilLib.getNewPath(path=self.data_dir,
                                        add='b')

        # a 파일 경로 모음
        self.files_a = [UtilLib.getNewPath(path=self.dir_a,
                                           add=filename)
                        for filename in os.listdir(self.dir_a)]
        # b 파일 경로 모음
        self.files_b = [UtilLib.getNewPath(path=self.dir_b,
                                           add=filename)
                        for filename in os.listdir(self.dir_b)]

        # 모드에 따른 데이터 전처리 방법
        self.transform = {
            const_var_cyclegan.MODE_TRAIN: transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(size=(const_var_cyclegan.RESIZE_SIZE, const_var_cyclegan.RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const_var_cyclegan.NORMALIZE_MEAN, std=const_var_cyclegan.NORMALIZE_STD)
            ]),
            const_var_cyclegan.MODE_TEST: transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(size=(const_var_cyclegan.RESIZE_SIZE, const_var_cyclegan.RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const_var_cyclegan.NORMALIZE_MEAN, std=const_var_cyclegan.NORMALIZE_STD)
            ])
        }

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, item):

        # 학습 모드시 데이터 학습 쌍을 다양하게 제공. 개수가 더 많은 데이터 쪽을 랜덤으로 뽑아 겹치는 데이터 학습 쌍을 최소화
        if self.mode_train_test == const_var_cyclegan.MODE_TRAIN:
            if len(self.files_a) >= len(self.files_b):
                # a 데이터 인덱스
                a_index = random.randint(0, len(self.files_a) - 1)
                # b 데이터 인덱스
                b_index = item % len(self.files_b)
            else:
                # a 데이터 인덱스
                a_index = item % len(self.files_a)
                # b 데이터 인덱스
                b_index = random.randint(0, len(self.files_b) - 1)

        # 테스트 모드시 데이터 학습 쌍을 일관되게 제공
        elif self.mode_train_test == const_var_cyclegan.MODE_TEST:
            # a 데이터 인덱스
            a_index = item % len(self.files_a)
            # b 데이터 인덱스
            b_index = item % len(self.files_b)

        # a 데이터
        a = self.transform[self.mode_train_test](Image.open(fp=self.files_a[a_index]))
        # b 데이터
        b = self.transform[self.mode_train_test](Image.open(fp=self.files_b[b_index]))

        return a, b
