import torch
import matplotlib.pyplot as plt

from Lib import UtilLib, DragonLib
from Common import const_var_lsgan


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


def save_checkpoint(filepath, modelG, modelD, optimizerG, optimizerD, epoch, is_best=False):
    """
    * 체크포인트 저장
    :param filepath: 저장될 체크포인트 파일 경로
    :param modelG: 저장될 모델. 생성자
    :param modelD: 저장될 모델. 판별자
    :param optimizerG: 저장될 optimizer. 생성자
    :param optimizerD: 저장될 optimizer. 판별자
    :param epoch: 저장될 현재 학습 epoch 횟수
    :param is_best: 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부
    :return: 체크포인트 파일 생성됨
    """

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # state 정보 담기
    state = {
        const_var_lsgan.KEY_STATE_MODEL_G: modelG.state_dict(),
        const_var_lsgan.KEY_STATE_MODEL_D: modelD.state_dict(),
        const_var_lsgan.KEY_STATE_OPTIMIZER_G: optimizerG.state_dict(),
        const_var_lsgan.KEY_STATE_OPTIMIZER_D: optimizerD.state_dict(),
        const_var_lsgan.KEY_STATE_EPOCH: epoch
    }

    # state 저장
    torch.save(obj=state,
               f=filepath)

    # 현재 저장하려는 모델이 가장 좋은 성능의 모델인 경우 best model 로 저장
    if is_best:
        torch.save(obj=state,
                   f=UtilLib.getNewPath(path=UtilLib.getParentDirPath(filePath=filepath),
                                        add=const_var_lsgan.CHECKPOINT_BEST_FILE_NAME))
