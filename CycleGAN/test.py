import os

import torch
from torchvision import transforms
from PIL import Image

from Common import const_var_cyclegan
from Lib import UtilLib, DragonLib


class Tester:
    def __init__(self, G, device):
        """
        * 테스트 관련 클래스
        :param G: 테스트 할 모델. 생성자
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.G = G
        # GPU / CPU
        self.device = device

    def running(self, input_dir, output_dir, generated_folder_name):
        """
        * 테스트 셋팅 및 진행
        :param input_dir: 입력 이미지 파일 디렉터리 위치
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param generated_folder_name: 생성된 이미지 파일 저장될 폴더명
        :return: 테스트 수행됨
        """

        # 테스트 진행
        result_list = self._test(input_dir=input_dir)

        # standardization 하는데 사용된 std, mean 값
        mean = torch.tensor(const_var_cyclegan.NORMALIZE_MEAN)
        std = torch.tensor(const_var_cyclegan.NORMALIZE_STD)

        for generated_img, img_filepath in result_list:

            # 시각화를 위해 standardization 한 거 원래대로 되돌리기, 값 범위 0 에서 1 로 제한 및 PIL image 로 변환
            generated_img_pil = self._convert_img(fake_x=generated_img, mean=mean, std=std)

            # 이미지 저장 경로
            img_filename = UtilLib.getOnlyFileName(filePath=img_filepath)
            generated_img_dir = UtilLib.getNewPath(path=output_dir, add=const_var_cyclegan.OUTPUT_DIR_SUFFIX_GENERATED_IMG.format(generated_folder_name))
            original_img_dir = UtilLib.getNewPath(path=generated_img_dir, add=const_var_cyclegan.OUTPUT_DIR_SUFFIX_ORIGINAL)
            transferred_img_dir = UtilLib.getNewPath(path=generated_img_dir, add=const_var_cyclegan.OUTPUT_DIR_SUFFIX_TRANSFERRED)
            original_img_filepath = UtilLib.getNewPath(path=original_img_dir, add=const_var_cyclegan.GENERATED_IMG_FILE_NAME.format(img_filename))
            transferred_img_filepath = UtilLib.getNewPath(path=transferred_img_dir, add=const_var_cyclegan.GENERATED_IMG_FILE_NAME.format(img_filename))

            # 원본 이미지 저장
            DragonLib.copyfile_with_make_parent_dir(src=img_filepath, dst=original_img_filepath)
            # 결과물 이미지 저장
            self._save_pics(fake_x_pil=generated_img_pil, filepath=transferred_img_filepath)

    def _test(self, input_dir):
        """
        * 테스트 진행
        :param input_dir: 입력 이미지 파일 디렉터리 위치
        :return: 이미지 생성 및 파일 경로 반환
        """

        # 생성한 이미지 및 원본 이미지 경로명 담을 리스트
        result_list = []

        # 모델을 테스트 모드로 전환
        self.G.eval()

        for image_filename in os.listdir(input_dir):

            # 이미지 읽고 변환하기
            img_filepath = UtilLib.getNewPath(path=input_dir, add=image_filename)
            img = self._read_img(filepath=img_filepath)
            img = img.unsqueeze(dim=0)

            # 이미지 생성
            img = img.to(self.device)
            generated_img = self.G(img)

            # 생성된 이미지와 원본 이미지 경로명 담기
            result_list.append((generated_img[0], img_filepath))

        return result_list

    @staticmethod
    def _read_img(filepath):
        """
        * 이미지 읽고 변환하기
        :param filepath: 읽어 올 이미지 파일 경로
        :return: 이미지 읽어 변환 해줌
        """

        # 데이터 변환 함수
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=(const_var_cyclegan.RESIZE_SIZE, const_var_cyclegan.RESIZE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=const_var_cyclegan.NORMALIZE_MEAN, std=const_var_cyclegan.NORMALIZE_STD)
        ])

        # 이미지 읽기 및 변환
        img = transform(Image.open(fp=filepath))

        return img

    @staticmethod
    def _convert_img(fake_x, mean, std):
        """
        * normalize (혹은 standardize) 된 데이터를 원래 데이터로 되돌리고 값 범위 0 에서 1 사이로 제한해주며 PIL image 로 바꿔주기
        :param fake_x: 생성된 이미지
        :param mean: mean 값
        :param std: std 값
        :return: 변환된 형태의 PIL image
        """

        # tensor 에서 PIL 로 변환시켜주는 함수
        transform = transforms.ToPILImage()

        # 정규화된 데이터 원래 데이터로 돌려놓기
        fake_x = fake_x.cpu().detach() * std[:, None, None] + mean[:, None, None]

        # 값의 범위를 0 에서 1 로 제한
        fake_x[fake_x > 1] = 1
        fake_x[fake_x < 0] = 0

        # PIL image 로 변환
        fake_x_pil = transform(fake_x)

        return fake_x_pil

    @staticmethod
    def _save_pics(fake_x_pil, filepath):
        """
        * 이미지 파일 저장
        :param fake_x_pil: 생성된 PIL image 형식의 이미지
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        fake_x_pil.save(fp=filepath)
