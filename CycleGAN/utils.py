import random

import torch
import matplotlib.pyplot as plt

from Lib import UtilLib, DragonLib
from Common import const_var_cyclegan


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


def save_checkpoint(filepath, G_AB, G_BA, D_A, D_B, optimizerG, optimizerD, epoch, is_best=False):
    """
    * 체크포인트 저장
    :param filepath: 저장될 체크포인트 파일 경로
    :param G_AB: 저장될 모델. 생성자
    :param G_BA: 저장될 모델. 생성자
    :param D_A: 저장될 모델. 판별자
    :param D_B: 저장될 모델. 판별자
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
        const_var_cyclegan.KEY_STATE_G_AB: G_AB.state_dict(),
        const_var_cyclegan.KEY_STATE_G_BA: G_BA.state_dict(),
        const_var_cyclegan.KEY_STATE_D_A: D_A.state_dict(),
        const_var_cyclegan.KEY_STATE_D_B: D_B.state_dict(),
        const_var_cyclegan.KEY_STATE_OPTIMIZER_G: optimizerG.state_dict(),
        const_var_cyclegan.KEY_STATE_OPTIMIZER_D: optimizerD.state_dict(),
        const_var_cyclegan.KEY_STATE_EPOCH: epoch
    }

    # state 저장
    torch.save(obj=state,
               f=filepath)

    # 현재 저장하려는 모델이 가장 좋은 성능의 모델인 경우 best model 로 저장
    if is_best:
        torch.save(obj=state,
                   f=UtilLib.getNewPath(path=UtilLib.getParentDirPath(filePath=filepath),
                                        add=const_var_cyclegan.CHECKPOINT_BEST_FILE_NAME))


def save_pics(pics_list, filepath, title):
    """
    * 학습 결과물 이미지로 저장
    :param pics_list: 원본, 재구성 이미지 쌍 담은 리스트
    :param filepath: 저장될 그림 파일 경로
    :param title: 그림 제목
    :return: 그림 파일 생성됨
    """

    # plt 로 시각화 할 수 있는 형식으로 변환
    mean = torch.tensor(const_var_cyclegan.NORMALIZE_MEAN)
    std = torch.tensor(const_var_cyclegan.NORMALIZE_STD)
    plt_pics_list = [(
        (a.cpu().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (b.cpu().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (fake_a.cpu().detach().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (fake_b.cpu().detach().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (rec_a.cpu().detach().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (rec_b.cpu().detach().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)
    ) for a, b, fake_a, fake_b, rec_a, rec_b in pics_list]

    # plt 에 그리기
    fig, axs = plt.subplots(nrows=const_var_cyclegan.NUM_PICS_LIST, ncols=6, figsize=(10, 15))
    fig.suptitle(t=title, fontsize=18)
    for num, (a, b, fake_a, fake_b, rec_a, rec_b) in enumerate(plt_pics_list):
        axs[num, 0].imshow(X=a, cmap='gray')
        axs[num, 0].axis('off')
        axs[num, 1].imshow(X=b, cmap='gray')
        axs[num, 1].axis('off')
        axs[num, 2].imshow(X=fake_a, cmap='gray')
        axs[num, 2].axis('off')
        axs[num, 3].imshow(X=fake_b, cmap='gray')
        axs[num, 3].axis('off')
        axs[num, 4].imshow(X=rec_a, cmap='gray')
        axs[num, 4].axis('off')
        axs[num, 5].imshow(X=rec_b, cmap='gray')
        axs[num, 5].axis('off')

        if num == 0:
            axs[num, 0].set_title('A')
            axs[num, 1].set_title('B')
            axs[num, 2].set_title('Fake A')
            axs[num, 3].set_title('Fake B')
            axs[num, 4].set_title('Reconstructed A')
            axs[num, 5].set_title('Reconstructed B')

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # 그림 저장
    plt.savefig(filepath)


class ReplayBuffer:
    def __init__(self, replay_buffer_max_size=50):
        """
        * replay buffer 관리 클래스
        :param replay_buffer_max_size: replay buffer 사이즈
        """

        # replay buffer 사이즈 1보다 작을 경우 에러 메세지 출력
        assert replay_buffer_max_size > 0, print(const_var_cyclegan.MSG_REPLAY_BUFFER_ERROR)

        # replay buffer 사이즈
        self.replay_buffer_max_size = replay_buffer_max_size
        # replay buffer 리스트
        self.replay_buffer = []

    def push_and_pop(self, fake_data_batch):
        """
        * replay buffer 에 저장된 데이터 혹은 입력으로 들어왔던 데이터 반환해주기
        :param fake_data_batch: 배치 개수 만큼의 생성된 fake data. (N, in_channels (3), H (256), W (256))
        :return: 배치 개수 만큼의 replay buffer 에 저장된 데이터 혹은 입력으로 들어왔던 데이터 (N, in_channels (3), H (256), W (256))
        """

        # 반환할 데이터 모아둔 리스트. 배치 개수 만큼의 (N 개의) 요소 담을 리스트
        data_to_return = []

        for fake_data in fake_data_batch:

            # (in_channels (3), H (256), W (256)) -> (1, in_channels (3), H (256), W (256))
            fake_data = torch.unsqueeze(input=fake_data.data, dim=0)

            # replay buffer 가 아직 가득 차있지 않은 경우 replay buffer 에 데이터 채우면서 들어오는 데이터 바로바로 반환하기
            if len(self.replay_buffer) < self.replay_buffer_max_size:
                self.replay_buffer.append(fake_data)
                data_to_return.append(fake_data)

            # replay buffer 가 가득찬 경우 본격적으로 현재 들어온 데이터를 바로 돌려줄지 replay buffer 에서 꺼내올지 50% 확률로 결정
            else:

                # replay buffer 에서 데이터 뽑아주기. replay buffer 에서 무작위로 데이터 하나 뽑아서 반환해주고 현재 데이터는 데이터 뽑아온 해당 replay buffer 위치에 대체해주기
                if random.randint(0, 1) == 1:
                    random_index = random.randint(0, self.replay_buffer_max_size - 1)
                    data_to_return.append(self.replay_buffer[random_index].clone())
                    self.replay_buffer[random_index] = fake_data

                # 그냥 현재 데이터 그대로 반환해주기
                else:
                    data_to_return.append(fake_data)

        return torch.cat(tensors=data_to_return, dim=0)
