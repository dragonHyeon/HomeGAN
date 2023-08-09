import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from Lib import UtilLib
from Common import const_var_lsgan


class HouseExpoDataset(Dataset):
    def __init__(self, data_dir, mode_train_test):
        """
        * HouseExpoDataset 데이터로더
        :param data_dir: 데이터 디렉터리
        :param mode_train_test: 학습 / 테스트 모드
        """

        # 데이터 해당 디렉터리
        self.data_dir = data_dir
        # 학습 / 테스트 모드
        self.mode_train_test = mode_train_test

        # 파일 경로 모음
        self.files = [UtilLib.getNewPath(path=data_dir,
                                         add=filename)
                      for filename in os.listdir(self.data_dir)]

        # 모드에 따른 데이터 전처리 방법
        self.transform = {
            const_var_lsgan.MODE_TRAIN: transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(size=const_var_lsgan.RESIZE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=const_var_lsgan.NORMALIZE_MEAN, std=const_var_lsgan.NORMALIZE_STD)
            ]),
            const_var_lsgan.MODE_TEST: transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(size=const_var_lsgan.RESIZE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=const_var_lsgan.NORMALIZE_MEAN, std=const_var_lsgan.NORMALIZE_STD)
            ])
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.transform[self.mode_train_test](Image.open(fp=self.files[item]))
