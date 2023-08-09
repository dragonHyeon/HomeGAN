import os, sys, argparse


def set_path():
    """
    * 경로 잡기
    """

    # 경로명 설정
    AppPath = os.path.dirname(os.path.abspath(os.getcwd()))
    SRC_DIR_NAME_Common = 'Common'
    SRC_DIR_NAME_DATA = 'DATA'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'
    SRC_DIR_NAME_SinGAN = 'SinGAN'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)
    SinGAN = os.path.join(AppPath, SRC_DIR_NAME_SinGAN)

    # 경로 추가
    AppPathList = [AppPath, Common, DATA, Lib, LOG, Main, RES, SinGAN]
    for p in AppPathList:
        sys.path.append(p)


def arguments():
    """
    * parser 이용하여 프로그램 실행 인자 받기
    :return: args
    """

    from Common import const_var_singan

    # parser 생성
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Train",
                                     description="* Run this to train the model.")

    # parser 인자 목록 생성
    # 학습 데이터 경로 설정
    parser.add_argument("--train_data_path",
                        type=str,
                        help='set training data path',
                        default=const_var_singan.DATA_PATH_TRAIN,
                        dest="train_data_path")

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=const_var_singan.OUTPUT_DIR,
                        dest='output_dir')

    # 체크포인트 파일 저장 및 학습 진행 기록 빈도수
    parser.add_argument("--tracking_frequency",
                        type=int,
                        help='set model training tracking frequency',
                        default=const_var_singan.TRACKING_FREQUENCY,
                        dest='tracking_frequency')

    # learning rate 설정
    parser.add_argument("--learning_rate",
                        type=float,
                        help='set learning rate',
                        default=const_var_singan.LEARNING_RATE,
                        dest='learning_rate')

    # 학습 반복 횟수
    parser.add_argument("--num_iter",
                        type=int,
                        help='set number of iterations to train',
                        default=const_var_singan.NUM_ITER,
                        dest='num_iter')

    # learning rate 감소 시작 할 iter 값
    parser.add_argument("--decay_iter_num",
                        type=int,
                        help='set iter num to start learning rate decay',
                        default=const_var_singan.DECAY_ITER_NUM,
                        dest='decay_iter_num')

    # parsing 한거 가져오기
    args = parser.parse_args()

    return args


def run_program(args):
    """
    * 학습 실행
    :param args: 프로그램 실행 인자
    :return: None
    """

    import torch

    from Common import const_var_singan
    from SinGAN import utils
    from SinGAN.train import Trainer
    from SinGAN.model import Generator, Discriminator
    from SinGAN.loss import loss_fn_GAN
    from SinGAN.metric import GAN_loss

    # GPU / CPU 설정
    device = const_var_singan.DEVICE_CUDA if torch.cuda.is_available() else const_var_singan.DEVICE_CPU

    # 이미지 피라미드 선언
    image_pyramids = utils.get_image_pyramids(filepath=args.train_data_path)

    # 모델 및 optimizer 선언
    Gs = []
    Ds = []
    optimizerGs = []
    optimizerDs = []
    in_channels = image_pyramids[0][const_var_singan.KEY_IMGPYR_IMAGE].shape[1]
    num_hidden = 16
    for current_scale_num in range(len(image_pyramids)):

        # 4 scale 마다 num_hidden 2 배로 증가
        if current_scale_num % 4 == 0:
            num_hidden *= 2

        G = Generator(channels=in_channels, num_hidden=num_hidden).to(device)
        D = Discriminator(in_channels=in_channels, num_hidden=num_hidden).to(device)

        Gs.append(G)
        Ds.append(D)

        optimizerG = torch.optim.Adam(params=G.parameters(),
                                      lr=args.learning_rate,
                                      betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(params=D.parameters(),
                                      lr=args.learning_rate,
                                      betas=(0.5, 0.999))

        optimizerGs.append(optimizerG)
        optimizerDs.append(optimizerD)

    # 모델 학습 객체 선언
    trainer = Trainer(Gs=Gs,
                      Ds=Ds,
                      optimizerGs=optimizerGs,
                      optimizerDs=optimizerDs,
                      loss_fn=None,
                      metric_fn=None,
                      image_pyramids=image_pyramids,
                      device=device)

    # 모델 학습
    trainer.running(num_iter=args.num_iter,
                    output_dir=args.output_dir,
                    train_data_path=args.train_data_path,
                    decay_iter_num=args.decay_iter_num,
                    tracking_frequency=args.tracking_frequency)


def main():

    # 경로 잡기
    set_path()

    # 인자 받기
    args = arguments()

    # 프로그램 실행
    run_program(args=args)


if __name__ == '__main__':
    main()
