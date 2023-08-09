import shutil

import visdom
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from Lib import UtilLib, DragonLib
from Common import const_var_singan
from SinGAN import utils


class Trainer:
    def __init__(self, Gs, Ds, optimizerGs, optimizerDs, loss_fn, metric_fn, image_pyramids, device):
        """
        * 학습 관련 클래스
        :param Gs: 학습 시킬 모델. 생성자 모음
        :param Ds: 학습 시킬 모델. 판별자 모음
        :param optimizerGs: 생성자 학습 optimizer 모음
        :param optimizerDs: 판별자 학습 optimizer 모음
        :param loss_fn: 손실 함수
        :param metric_fn: 성능 평가 지표
        :param image_pyramids: 이미지 피라미드
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.Gs = Gs
        self.Ds = Ds
        # 학습 optimizer
        self.optimizerGs = optimizerGs
        self.optimizerDs = optimizerDs
        # 손실 함수
        self.loss_fn = loss_fn
        # 이미지 피라미드
        self.image_pyramids = image_pyramids
        # GPU / CPU
        self.device = device

        # 고정 노이즈 담을 리스트
        self.fixed_zs = []
        # 시그마 담을 리스트
        self.sigmas = []

    def running(self, num_iter, output_dir, train_data_path, decay_iter_num, tracking_frequency):
        """
        * 학습 셋팅 및 진행
        :param num_iter: 학습 반복 횟수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param train_data_path: 학습한 해당 이미지 경로 (체크포인트 저장시 사용)
        :param decay_iter_num: learning rate 줄어들기 시작하는 iter 수
        :param tracking_frequency: 체크포인트 파일 저장 및 학습 진행 기록 빈도수
        :return: 학습 완료 및 체크포인트 파일 생성됨
        """

        # 체크포인트 저장 관련 경로 및 디렉터리
        image_name = UtilLib.getOnlyFileName(filePath=train_data_path)
        checkpoint_dir = UtilLib.getNewPath(path=output_dir,
                                            add=const_var_singan.OUTPUT_DIR_SUFFIX_CHECKPOINT.format(image_name))
        checkpoint_iter_dir = UtilLib.getNewPath(path=checkpoint_dir,
                                                 add=const_var_singan.OUTPUT_DIR_SUFFIX_CHECKPOINT_ITER.format(num_iter))
        # 체크포인트 저장 경로
        checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_iter_dir,
                                                 add=const_var_singan.CHECKPOINT_FILE_NAME)
        # 체크포인트 학습 이미지 저장 경로
        checkpoint_train_img_filepath = UtilLib.getNewPath(path=checkpoint_dir,
                                                           add=const_var_singan.CHECKPOINT_TRAIN_IMG_FILE_NAME)

        # 각 scale 별로 모델 학습 진행
        for current_scale_num, (G, D) in enumerate(zip(self.Gs, self.Ds)):

            # iter 초기화
            start_iter_num = const_var_singan.INITIAL_START_ITER_NUM

            # 각 모델 가중치 초기화
            try:
                G.load_state_dict(G[current_scale_num - 1].state_dict())
                D.load_state_dict(D[current_scale_num - 1].state_dict())
            except:
                # G.apply(weights_init)
                # D.apply(weights_init)
                pass

            # optimizer learning rate 스케쥴러 선언
            lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerGs[current_scale_num],
                                                                  milestones=[1600],
                                                                  gamma=0.1)
            lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerDs[current_scale_num],
                                                                  milestones=[1600],
                                                                  gamma=0.1)

            # num iter 만큼 학습 반복
            for current_iter_num in tqdm(range(start_iter_num, num_iter + 1),
                                         desc='training process ({0}/{1})'.format(current_scale_num + 1, len(self.image_pyramids)),
                                         total=num_iter,
                                         initial=start_iter_num - 1):

                # 학습 진행
                fixed_z, sigma = self._train(current_scale_num=current_scale_num)

                # 학습 진행 기록 주기마다 학습 진행 시각화
                if current_iter_num % tracking_frequency == 0:

                    # 테스트 진행
                    score, fake_x = self._eval(current_scale_num=current_scale_num)

                    # 그래프 시각화 진행
                    self._draw_graph(score=score,
                                     current_iter_num=current_iter_num,
                                     title='Loss Progress ({0}/{1})'.format(current_scale_num + 1, len(self.image_pyramids)))

                # learning rate 업데이트
                lr_scheduler_G.step()
                lr_scheduler_D.step()

            # 학습을 통해 나온 고정 노이즈와 시그마 담기
            self.fixed_zs.append(fixed_z)
            self.sigmas.append(sigma)

            # 결과물 시각화 진행
            # 체크포인트 결과물 이미지 저장 경로
            checkpoint_result_img_filepath = UtilLib.getNewPath(path=checkpoint_iter_dir,
                                                                add=const_var_singan.CHECKPOINT_RESULT_IMG_FILE_NAME.format(current_scale_num + 1, len(self.image_pyramids)))
            self._draw_pic(fake_x=fake_x,
                           x=self.image_pyramids[current_scale_num][const_var_singan.KEY_IMGPYR_IMAGE],
                           mean=self.image_pyramids[current_scale_num][const_var_singan.KEY_IMGPYR_MEAN],
                           std=self.image_pyramids[current_scale_num][const_var_singan.KEY_IMGPYR_STD],
                           current_scale_num=current_scale_num,
                           filepath=checkpoint_result_img_filepath)

            # 현재 모델에 대한 학습 끝나면 다음 모델에 대한 새로운 학습 그래프 그리기 위해 현재 plt 는 지우기
            del self.plt

        # 체크포인트 저장
        utils.save_checkpoint(filepath=checkpoint_filepath,
                              Gs=self.Gs,
                              Ds=self.Ds,
                              sigmas=self.sigmas,
                              image_pyramids=self.image_pyramids,
                              is_best=False)

        # 체크포인트 저장시 학습에 사용된 이미지도 함께 저장
        shutil.copyfile(src=train_data_path, dst=checkpoint_train_img_filepath)

    def _train(self, current_scale_num):
        """
        * 학습 진행
        :param current_scale_num: 현재 학습 시킬 scale
        :return: 1 iter 만큼 학습 진행
        """

        # 학습 시킬 현재 scale 의 모델
        G = self.Gs[current_scale_num]
        D = self.Ds[current_scale_num]

        # 각 모델을 학습 모드로 전환
        G.train()
        D.train()

        mse = nn.MSELoss()
        init_sigma = 0.1

        # 학습 시킬 현재 scale 의 이미지
        x = self.image_pyramids[current_scale_num][const_var_singan.KEY_IMGPYR_IMAGE].to(self.device)
        # 학습 시킬 현재 scale 의 고정 노이즈
        fixed_z = torch.randn(size=x.shape).to(self.device) if current_scale_num == 0 else torch.zeros_like(input=x).to(self.device)
        # 학습 시킬 현재 scale 의 이전 단계 결과물
        prev = torch.zeros_like(input=x).to(self.device) if current_scale_num == 0 else self._get_prev(current_scale_num=current_scale_num, is_fixed=False, mode_train_test=const_var_singan.MODE_TRAIN).to(self.device).detach()
        # 학습 시킬 현재 scale 의 이전 단계 결과물 (고정 노이즈 사용)
        fixed_prev = torch.zeros_like(input=x).to(self.device) if current_scale_num == 0 else self._get_prev(current_scale_num=current_scale_num, is_fixed=True, mode_train_test=const_var_singan.MODE_TRAIN).to(self.device).detach()
        # 학습 시킬 현재 scale 의 시그마 값
        sigma = 1 if current_scale_num == 0 else torch.sqrt(mse(fixed_prev, x)) * init_sigma
        # 학습 시킬 현재 scale 의 입력 노이즈
        z = torch.randn(size=x.shape).to(self.device) if current_scale_num == 0 else torch.randn(size=x.shape).to(self.device) * sigma

        # ----------------------
        #  Train Discriminators
        # ----------------------
        for j in range(3):

            # GAN loss
            fake_x = G(z, prev)
            fake_logits = D(fake_x.detach())
            real_logits = D(x)
            loss_D_GAN = fake_logits.mean() - real_logits.mean() + gradient_penalty(x, fake_x, D, lambda_=0.1)

            # Total loss
            loss_D = loss_D_GAN

            # 역전파
            self.optimizerDs[current_scale_num].zero_grad()
            loss_D.backward()
            self.optimizerDs[current_scale_num].step()

        # ------------------
        #  Train Generators
        # ------------------
        for j in range(3):

            # GAN loss
            fake_x = G(z, prev)
            fake_logits = D(fake_x)
            loss_G_GAN = -fake_logits.mean()

            # Rec loss
            fixed_fake_x = G(fixed_z, fixed_prev)
            loss_G_rec = mse(fixed_fake_x, x)

            # Total loss
            loss_G = loss_G_GAN + const_var_singan.LAMBDA_REC * loss_G_rec

            # 역전파
            self.optimizerGs[current_scale_num].zero_grad()
            loss_G.backward()
            self.optimizerGs[current_scale_num].step()

        return fixed_z, sigma

    def _eval(self, current_scale_num):
        """
        * 테스트 진행
        :param current_scale_num: 현재 테스트 할 scale
        :return: 현재 scale 성능 평가 점수, 학습 결과물
        """

        # 테스트 할 현재 scale 의 모델
        G = self.Gs[current_scale_num]
        D = self.Ds[current_scale_num]

        # 각 모델을 테스트 모드로 전환
        G.eval()
        D.eval()

        mse = nn.MSELoss()
        init_sigma = 0.1

        # 테스트 할 현재 scale 의 이미지
        x = self.image_pyramids[current_scale_num][const_var_singan.KEY_IMGPYR_IMAGE].to(self.device)
        # 테스트 할 현재 scale 의 고정 노이즈
        fixed_z = torch.randn(size=x.shape).to(self.device) if current_scale_num == 0 else torch.zeros_like(input=x).to(self.device)
        # 테스트 할 현재 scale 의 이전 단계 결과물
        prev = torch.zeros_like(input=x).to(self.device) if current_scale_num == 0 else self._get_prev(current_scale_num=current_scale_num, is_fixed=False, mode_train_test=const_var_singan.MODE_TEST).to(self.device).detach()
        # 테스트 할 현재 scale 의 이전 단계 결과물 (고정 노이즈 사용)
        fixed_prev = torch.zeros_like(input=x).to(self.device) if current_scale_num == 0 else self._get_prev(current_scale_num=current_scale_num, is_fixed=True, mode_train_test=const_var_singan.MODE_TEST).to(self.device).detach()
        # 테스트 할 현재 scale 의 시그마 값
        sigma = 1 if current_scale_num == 0 else torch.sqrt(mse(fixed_prev, x)) * init_sigma
        # 테스트 할 현재 scale 의 입력 노이즈
        z = torch.randn(size=x.shape).to(self.device) if current_scale_num == 0 else torch.randn(size=x.shape).to(self.device) * sigma

        # ---------------------
        #  Test Discriminators
        # ---------------------

        # GAN loss
        fake_x = G(z, prev)
        fake_logits = D(fake_x.detach())
        real_logits = D(x)
        loss_D_GAN = fake_logits.mean() - real_logits.mean() + gradient_penalty(x, fake_x, D, lambda_=0.1)

        # Total loss
        loss_D = loss_D_GAN

        # -----------------
        #  Test Generators
        # -----------------

        # GAN loss
        fake_x = G(z, prev)
        fake_logits = D(fake_x)
        loss_G_GAN = -fake_logits.mean()

        # Rec loss
        fixed_fake_x = G(fixed_z, fixed_prev)
        loss_G_rec = mse(fixed_fake_x, x)

        # Total loss
        loss_G = loss_G_GAN + const_var_singan.LAMBDA_REC * loss_G_rec

        # score 기록
        score = {
            const_var_singan.KEY_SCORE_D: loss_D,
            const_var_singan.KEY_SCORE_G: loss_G
        }

        return score, fake_x

    def _get_prev(self, current_scale_num, is_fixed, mode_train_test):
        """
        * 학습 시킬 현재 scale 의 이전 단계 결과물 가져오기
        :param current_scale_num: 학습 시킬 현재 scale
        :param is_fixed: 고정 노이즈 사용 여부
        :param mode_train_test: 학습 / 테스트 모드
        :return: upscale 한 이전 단계 결과물
        """

        # 이전 단계 결과물을 현재 scale 로 키우기 위한 함수
        upsample = nn.Upsample(size=self.image_pyramids[current_scale_num][const_var_singan.KEY_IMGPYR_IMAGE].shape[2:])

        # 이전 단계 변수 설정
        prev_scale_num = current_scale_num - 1
        prev_G = self.Gs[prev_scale_num]
        if mode_train_test == const_var_singan.MODE_TRAIN:
            prev_G.train()
        elif mode_train_test == const_var_singan.MODE_TEST:
            prev_G.eval()
        prev_x = self.image_pyramids[prev_scale_num][const_var_singan.KEY_IMGPYR_IMAGE]
        prev_sigma = self.sigmas[prev_scale_num]
        if is_fixed:
            prev_z = self.fixed_zs[prev_scale_num].to(self.device)
        else:
            prev_z = torch.randn(size=prev_x.shape).to(self.device) if prev_scale_num == 0 else torch.randn(size=prev_x.shape).to(self.device) * prev_sigma

        # 재귀적으로 이전 단계 결과물 가져오기
        if prev_scale_num > 0:
            return upsample(prev_G(prev_z, self._get_prev(current_scale_num=prev_scale_num, is_fixed=is_fixed, mode_train_test=mode_train_test)))
        # 가장 첫 단계의 결과물 가져오기 (논문에서는 N 단계)
        else:
            return upsample(prev_G(prev_z, torch.zeros_like(input=prev_x).to(self.device)))

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
                                                 add=const_var_singan.CHECKPOINT_BEST_FILE_NAME)
            # 기존에 측정한 best 체크포인트가 있으면 해당 score 로 초기화
            if UtilLib.isExist(checkpoint_file):
                state = utils.load_checkpoint(filepath=checkpoint_file)
                self.best_score = state[const_var_singan.KEY_SCORE]
            # 없다면 임의의 큰 숫자 (100000) 로 초기화
            else:
                self.best_score = const_var_singan.INITIAL_BEST_SCORE

        # best 성능 갱신
        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def _draw_graph(self, score, current_iter_num, title):
        """
        * 학습 진행 상태 실시간으로 시각화
        :param score: 성능 평가 점수
        :param current_iter_num: 현재 학습 수
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
            self.vis.line(Y=torch.cat((torch.Tensor([score[const_var_singan.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[const_var_singan.KEY_SCORE_D]]).view(-1, 1)),
                                      dim=1),
                          X=torch.cat((torch.Tensor([current_iter_num]).view(-1, 1), torch.Tensor([current_iter_num]).view(-1, 1)),
                                      dim=1),
                          win=self.plt,
                          update='append',
                          opts=dict(title=title,
                                    legend=['G loss', 'D loss'],
                                    showlegend=True))
        except AttributeError:
            self.plt = self.vis.line(Y=torch.cat((torch.Tensor([score[const_var_singan.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[const_var_singan.KEY_SCORE_D]]).view(-1, 1)),
                                                 dim=1),
                                     X=torch.cat((torch.Tensor([current_iter_num]).view(-1, 1), torch.Tensor([current_iter_num]).view(-1, 1)),
                                                 dim=1),
                                     opts=dict(title=title,
                                               legend=['G loss', 'D loss'],
                                               showlegend=True))

    def _draw_pic(self, fake_x, x, mean, std, current_scale_num, filepath):
        """
        * 학습 결과물 이미지로 저장
        :param fake_x: 생성된 이미지
        :param x: 원본 이미지
        :param mean: 해당 scale 원본 이미지의 mean 값
        :param std: 해당 scale 원본 이미지의 std 값
        :param current_scale_num: 현재 scale
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 시각화를 위해 standardization 한 거 원래대로 되돌리기
        # plt 에 맞게 (N (1), C, H, W) -> (H, W, C) 변환
        fake_x_plt = (fake_x[0].cpu().detach() * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)
        x_plt = (x[0] * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)

        # 시각화 진행
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        fig.suptitle(t='current scale: {0} (total scale: {1})'.format(current_scale_num + 1, len(self.image_pyramids)), fontsize=18)
        axs[0].imshow(X=fake_x_plt, cmap='gray')
        axs[0].set_title('fake_x')
        axs[0].axis('off')
        axs[1].imshow(X=x_plt, cmap='gray')
        axs[1].set_title('x')
        axs[1].axis('off')

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        plt.savefig(filepath)


def gradient_penalty(x_real, x_fake, D, lambda_=10):
    eps = torch.rand(1, 1, 1, 1).to("cuda:0")
    x_hat = eps * x_real + (1. - eps) * x_fake
    x_hat = autograd.Variable(x_hat, requires_grad=True)
    outputs = D(x_hat)
    grads = autograd.grad(outputs, x_hat, torch.ones_like(outputs), retain_graph=True, create_graph=True)[0]
    penalty = lambda_ * ((torch.norm(grads, p=2, dim=1) - 1) ** 2).mean()
    return penalty


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
