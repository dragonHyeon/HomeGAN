import visdom
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from Lib import UtilLib, DragonLib
from Common import const_var_lsgan
from LSGAN import utils


class Trainer:
    def __init__(self, modelG, modelD, optimizerG, optimizerD, loss_fn, metric_fn, train_dataloader, test_dataloader, device):
        """
        * 학습 관련 클래스
        :param modelG: 학습 시킬 모델. 생성자
        :param modelD: 학습 시킬 모델. 판별자
        :param optimizerG: 생성자 학습 optimizer
        :param optimizerD: 판별자 학습 optimizer
        :param loss_fn: 손실 함수
        :param metric_fn: 성능 평가 지표
        :param train_dataloader: 학습용 데이터로더
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.modelG = modelG
        self.modelD = modelD
        # 학습 optimizer
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        # 손실 함수
        self.loss_fn = loss_fn
        # 성능 평가 지표
        self.metric_fn = metric_fn
        # 학습용 데이터로더
        self.train_dataloader = train_dataloader
        # 테스트용 데이터로더
        self.test_dataloader = test_dataloader
        # GPU / CPU
        self.device = device

    def running(self, num_epoch, output_dir, tracking_frequency, checkpoint_file=None):
        """
        * 학습 셋팅 및 진행
        :param num_epoch: 학습 반복 횟수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param tracking_frequency: 체크포인트 파일 저장 및 학습 진행 기록 빈도수
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 학습 완료 및 체크포인트 파일 생성됨
        """

        # 학습 중간 중간 생성자로 이미지를 생성하기 위한 샘플 noise z 모음
        sample_z_collection = torch.randn(size=(20, 100, 1, 1), device=self.device)

        # epoch 초기화
        start_epoch_num = const_var_lsgan.INITIAL_START_EPOCH_NUM

        # 각 모델 가중치 초기화
        self.modelG.apply(weights_init)
        self.modelD.apply(weights_init)

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.modelG.load_state_dict(state[const_var_lsgan.KEY_STATE_MODEL_G])
            self.modelD.load_state_dict(state[const_var_lsgan.KEY_STATE_MODEL_D])
            self.optimizerG.load_state_dict(state[const_var_lsgan.KEY_STATE_OPTIMIZER_G])
            self.optimizerD.load_state_dict(state[const_var_lsgan.KEY_STATE_OPTIMIZER_D])
            start_epoch_num = state[const_var_lsgan.KEY_STATE_EPOCH] + 1

        # num epoch 만큼 학습 반복
        for current_epoch_num in tqdm(range(start_epoch_num, num_epoch + 1),
                                      desc='training process',
                                      total=num_epoch,
                                      initial=start_epoch_num - 1):

            # 학습 진행
            self._train()

            # 학습 진행 기록 주기마다 학습 진행 저장 및 시각화
            if current_epoch_num % tracking_frequency == 0:

                # 테스트 진행
                score, generated_image_collection = self._eval(sample_z_collection=sample_z_collection)

                # 체크포인트 저장
                checkpoint_dir = UtilLib.getNewPath(path=output_dir, add=const_var_lsgan.OUTPUT_DIR_SUFFIX_CHECKPOINT)
                checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_dir, add=const_var_lsgan.CHECKPOINT_FILE_NAME.format(current_epoch_num))
                utils.save_checkpoint(filepath=checkpoint_filepath,
                                      modelG=self.modelG,
                                      modelD=self.modelD,
                                      optimizerG=self.optimizerG,
                                      optimizerD=self.optimizerD,
                                      epoch=current_epoch_num,
                                      is_best=False)

                # 그래프 시각화 진행
                self._draw_graph(score=score,
                                 current_epoch_num=current_epoch_num,
                                 title=nn.MSELoss.__name__)

                # 결과물 시각화 진행
                pics_dir = UtilLib.getNewPath(path=output_dir, add=const_var_lsgan.OUTPUT_DIR_SUFFIX_PICS)
                pics_filepath = UtilLib.getNewPath(path=pics_dir, add=const_var_lsgan.PICS_FILE_NAME.format(current_epoch_num))
                self._draw_pic(generated_image_collection=generated_image_collection,
                               title='Epoch {0}'.format(current_epoch_num),
                               filepath=pics_filepath)

    def _train(self):
        """
        * 학습 진행
        :return: 1 epoch 만큼 학습 진행
        """

        # 각 모델을 학습 모드로 전환
        self.modelG.train()
        self.modelD.train()

        # loss 선언
        mse = nn.MSELoss()

        # x shape: (N, 3, 64, 64)
        for x in tqdm(self.train_dataloader, desc='train dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = x.shape[0]

            # real image label
            real_label = torch.ones(batch_size, device=self.device)
            # fake image label
            fake_label = torch.zeros(batch_size, device=self.device)

            # noise z
            z = torch.randn(size=(batch_size, 100, 1, 1), device=self.device)

            # 텐서를 해당 디바이스로 이동
            x = x.to(self.device)

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # GAN loss
            loss_D_GAN_real = mse(self.modelD(x), real_label)
            fake_x = self.modelG(z)
            loss_D_GAN_fake = mse(self.modelD(fake_x.detach()), fake_label)
            loss_D_GAN = (loss_D_GAN_real + loss_D_GAN_fake) / 2

            # Total loss
            loss_D = loss_D_GAN

            # 역전파
            self.modelD.zero_grad()
            loss_D.backward()
            self.optimizerD.step()

            # ------------------
            #  Train Generators
            # ------------------

            # GAN loss
            fake_x = self.modelG(z)
            loss_G_GAN = mse(self.modelD(fake_x), real_label)

            # Total loss
            loss_G = loss_G_GAN

            # 역전파
            self.modelG.zero_grad()
            loss_G.backward()
            self.optimizerG.step()

    def _eval(self, sample_z_collection):
        """
        * 테스트 진행
        :param sample_z_collection: 생성자로 이미지를 생성하기 위한 샘플 noise z 모음
        :return: 현재 epoch 성능 평가 점수, 학습 결과물
        """

        # 각 모델을 테스트 모드로 전환
        self.modelG.eval()
        self.modelD.eval()

        # loss 선언
        mse = nn.MSELoss()

        # 배치 마다의 score 담을 리스트
        batch_score_listG = list()
        batch_score_listD = list()

        # x shape: (N (1), 3, 64, 64)
        for x in tqdm(self.test_dataloader, desc='test dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = x.shape[0]

            # real image label
            real_label = torch.ones(batch_size, device=self.device)
            # fake image label
            fake_label = torch.zeros(batch_size, device=self.device)

            # noise z
            z = torch.randn(size=(batch_size, 100, 1, 1), device=self.device)

            # 텐서를 해당 디바이스로 이동
            x = x.to(self.device)

            # ---------------------
            #  Test Discriminators
            # ---------------------

            # GAN loss
            score_D_GAN_real = mse(self.modelD(x), real_label)
            fake_x = self.modelG(z)
            score_D_GAN_fake = mse(self.modelD(fake_x), fake_label)
            score_D_GAN = (score_D_GAN_real + score_D_GAN_fake) / 2

            # Total loss
            score_D = score_D_GAN

            # batch score 담기
            batch_score_listD.append(score_D.cpu().detach())

            # -----------------
            #  Test Generators
            # -----------------

            # GAN loss
            fake_x = self.modelG(z)
            # 배치 마다의 생성자 MSE loss 계산
            score_G_GAN = mse(self.modelD(fake_x), real_label)

            # Total loss
            score_G = score_G_GAN

            # batch score 담기
            batch_score_listG.append(score_G.cpu().detach())

        # score 기록
        score = {
            const_var_lsgan.KEY_SCORE_G: np.mean(batch_score_listG),
            const_var_lsgan.KEY_SCORE_D: np.mean(batch_score_listD)
        }

        # 샘플 noise z 모음으로 이미지 생성하기
        generated_image_collection = self.modelG(sample_z_collection)

        return score, generated_image_collection

    def _check_is_best(self, score, best_checkpoint_dir):
        """
        * 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부 확인
        :param score: 현재 모델의 성능 점수
        :param best_checkpoint_dir: 비교할 best 체크포인트 파일 디렉터리 위치
        :return: True / False
        """

        # best 성능 측정을 위해 초기화
        try:
            self.best_score
        except AttributeError:
            checkpoint_file = UtilLib.getNewPath(path=best_checkpoint_dir,
                                                 add=const_var_lsgan.CHECKPOINT_BEST_FILE_NAME)
            # 기존에 측정한 best 체크포인트가 있으면 해당 score 로 초기화
            if UtilLib.isExist(checkpoint_file):
                state = utils.load_checkpoint(filepath=checkpoint_file)
                self.best_score = state[const_var_lsgan.KEY_SCORE]
            # 없다면 임의의 큰 숫자 (100000) 로 초기화
            else:
                self.best_score = const_var_lsgan.INITIAL_BEST_BCE_LOSS

        # best 성능 갱신
        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def _draw_graph(self, score, current_epoch_num, title):
        """
        * 학습 진행 상태 실시간으로 시각화
        :param score: 성능 평가 점수
        :param current_epoch_num: 현재 에폭 수
        :param title: 그래프 제목
        :return: visdom 으로 시각화 진행
        """

        # 서버 켜기
        try:
            self.vis
        except AttributeError:
            self.vis = visdom.Visdom()
        # 실시간으로 학습 진행 상태 그리기
        try:
            self.vis.line(Y=torch.cat((torch.Tensor([score[const_var_lsgan.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[const_var_lsgan.KEY_SCORE_D]]).view(-1, 1)), dim=1),
                          X=torch.cat((torch.Tensor([current_epoch_num]).view(-1, 1), torch.Tensor([current_epoch_num]).view(-1, 1)), dim=1),
                          win=self.plt,
                          update='append',
                          opts=dict(title=title,
                                    legend=['G loss', 'D loss'],
                                    showlegend=True))
        except AttributeError:
            self.plt = self.vis.line(Y=torch.cat((torch.Tensor([score[const_var_lsgan.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[const_var_lsgan.KEY_SCORE_D]]).view(-1, 1)), dim=1),
                                     X=torch.cat((torch.Tensor([current_epoch_num]).view(-1, 1), torch.Tensor([current_epoch_num]).view(-1, 1)), dim=1),
                                     opts=dict(title=title,
                                               legend=['G loss', 'D loss'],
                                               showlegend=True))

    def _draw_pic(self, generated_image_collection, title, filepath):
        """
        * 학습 결과물 이미지로 저장
        :param generated_image_collection: 생성된 이미지 모음
        :param title: 그림 제목
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 시각화를 위해 standardization 한 거 원래대로 되돌리기
        # plt 에 맞게 (N, C, H, W) -> (N, H, W, C) 변환
        mean = torch.tensor(const_var_lsgan.NORMALIZE_MEAN)
        std = torch.tensor(const_var_lsgan.NORMALIZE_STD)
        plt_pics_list = [(generated_image.cpu().detach().reshape(-1, const_var_lsgan.RESIZE_SIZE, const_var_lsgan.RESIZE_SIZE) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0) for generated_image in generated_image_collection]

        # 시각화 진행
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(15, 10))
        fig.suptitle(t=title, fontsize=18)
        axs = axs.flatten()
        for num, generated_image in enumerate(plt_pics_list):
            axs[num].imshow(X=generated_image, cmap='gray')
            axs[num].axis('off')

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        plt.savefig(filepath)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
