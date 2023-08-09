import random

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from Lib import UtilLib, DragonLib
from Common import const_var_singan


def load_checkpoint(filepath):
    """
    * 체크포인트 불러오기
    :param filepath: 불러올 체크포인트 파일 경로
    :return: state 모음 (model.state_dict(), optimizer.state_dict(), epoch)
    """

    # state 불러오기
    state = torch.load(f=filepath)

    # state 정보 리턴
    return state


def save_checkpoint(filepath, Gs, Ds, sigmas, image_pyramids, is_best=False):
    """
    * 체크포인트 저장
    :param filepath: 저장될 체크포인트 파일 경로
    :param Gs: 저장될 모델. 생성자 모음
    :param Ds: 저장될 모델. 판별자 모음
    :param sigmas: 시그마 모음
    :param image_pyramids: 이미지 피라미드
    :param is_best: 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부
    :return: 체크포인트 파일 생성됨
    """

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # state 정보 담기
    state = {
        const_var_singan.KEY_STATE_GS: Gs,
        const_var_singan.KEY_STATE_DS: Ds,
        const_var_singan.KEY_STATE_SIGMAS: sigmas,
        const_var_singan.KEY_STATE_IMAGE_PYRAMIDS: image_pyramids
    }

    # state 저장
    torch.save(obj=state,
               f=filepath)

    # 현재 저장하려는 모델이 가장 좋은 성능의 모델인 경우 best model 로 저장
    if is_best:
        torch.save(obj=state,
                   f=UtilLib.getNewPath(path=UtilLib.getParentDirPath(filePath=filepath),
                                        add=const_var_singan.CHECKPOINT_BEST_FILE_NAME))


def get_image_pyramids(filepath, max_value=64, min_value=25, scale_factor=4/3):
    """
    * 이미지 피라미드 만들기
    :param filepath: 이미지 피라미드 만들 원본 이미지 경로
    :param max_value: 이미지 피라미드 가장 큰 스케일 크기
    :param min_value: 이미지 피라미드 가장 작은 스케일 크기
    :param scale_factor: 이미지 피라미드 변환 스케일
    :return: tensor 이미지 피라미드, 해당 scale 에서의 mean, std 값 담은 dict 만들어서 반환
    """

    # 원본 이미지 열기
    image = Image.open(fp=filepath)
    width, height = image.size

    # 이미지 피라미드 내 최대, 최소 길이 비율 계산
    if width >= height:
        max_w = max_value
        max_h = max_value * height / width
        min_w = min_value * width / height
        min_h = min_value
    else:
        max_w = max_value * width / height
        max_h = max_value
        min_w = min_value
        min_h = min_value * height / width

    # 스케일 목록 담기
    width_list = []
    height_list = []
    scale_num = 0
    while max(min_w, min_h) * scale_factor ** scale_num < max_value:
        width_list.append(min_w * scale_factor ** scale_num)
        height_list.append(min_h * scale_factor ** scale_num)
        scale_num += 1
    width_list.append(max_w)
    height_list.append(max_h)

    # 이미지 피라미드 담기
    image_pyramids = []
    for w, h in zip(width_list, height_list):

        # 데이터 전처리 (resize, normalization)
        transform = transforms.Compose([
            transforms.Resize(size=(int(h), int(w))),
            transforms.ToTensor(),
        ])
        transformed_img = transform(image)

        # 데이터 전처리 (standardization)
        mean = torch.mean(input=transformed_img, dim=(1, 2))
        std = torch.std(input=transformed_img, dim=(1, 2))
        transform = transforms.Normalize(mean=mean, std=std)
        transformed_img = transform(transformed_img)

        # 데이터 담기
        image_pyramids.append({
            const_var_singan.KEY_IMGPYR_IMAGE: torch.unsqueeze(input=transformed_img, dim=0),
            const_var_singan.KEY_IMGPYR_MEAN: mean,
            const_var_singan.KEY_IMGPYR_STD: std
        })

    return image_pyramids
