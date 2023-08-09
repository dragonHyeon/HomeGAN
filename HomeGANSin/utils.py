import torch


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
