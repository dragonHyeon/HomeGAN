import os, sys, argparse, itertools


def set_path():
    """
    * 경로 잡기
    """

    # 경로명 설정
    AppPath = os.path.dirname(os.path.abspath(os.getcwd()))
    SRC_DIR_NAME_Common = 'Common'
    SRC_DIR_NAME_CycleGAN = 'CycleGAN'
    SRC_DIR_NAME_DATA = 'DATA'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    CycleGAN = os.path.join(AppPath, SRC_DIR_NAME_CycleGAN)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)

    # 경로 추가
    AppPathList = [AppPath, Common, CycleGAN, DATA, Lib, LOG, Main, RES]
    for p in AppPathList:
        sys.path.append(p)


def arguments():
    """
    * parser 이용하여 프로그램 실행 인자 받기
    :return: args
    """

    from Common import const_var_cyclegan

    # parser 생성
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Train",
                                     description="* Run this to train the model.")

    # parser 인자 목록 생성
    # 학습 데이터 디렉터리 설정
    parser.add_argument("--train_data_dir",
                        type=str,
                        help='set training data directory',
                        default=const_var_cyclegan.CYCLEGAN_DIR_TRAIN,
                        dest="train_data_dir")

    # 테스트 데이터 디렉터리 설정
    parser.add_argument("--test_data_dir",
                        type=str,
                        help='set test data directory',
                        default=const_var_cyclegan.CYCLEGAN_DIR_TEST,
                        dest="test_data_dir")

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=const_var_cyclegan.OUTPUT_DIR,
                        dest='output_dir')

    # 체크포인트 파일 저장 및 학습 진행 기록 빈도수
    parser.add_argument("--tracking_frequency",
                        type=int,
                        help='set model training tracking frequency',
                        default=const_var_cyclegan.TRACKING_FREQUENCY,
                        dest='tracking_frequency')

    # 불러올 체크포인트 파일 경로
    parser.add_argument("--checkpoint_file",
                        type=str,
                        help='set checkpoint file to resume training if exists',
                        default=None,
                        dest='checkpoint_file')

    # learning rate 설정
    parser.add_argument("--learning_rate",
                        type=float,
                        help='set learning rate',
                        default=const_var_cyclegan.LEARNING_RATE,
                        dest='learning_rate')

    # 배치 사이즈
    parser.add_argument("--batch_size",
                        type=int,
                        help='set batch size',
                        default=const_var_cyclegan.BATCH_SIZE,
                        dest='batch_size')

    # 학습 반복 횟수
    parser.add_argument("--num_epoch",
                        type=int,
                        help='set number of epochs to train',
                        default=const_var_cyclegan.NUM_EPOCH,
                        dest='num_epoch')

    # dataloader 에서 데이터 불러오기 시 shuffle 여부
    parser.add_argument("--shuffle",
                        type=bool,
                        help='set whether to shuffle or not while loading dataloader',
                        default=const_var_cyclegan.SHUFFLE,
                        dest='shuffle')

    # learning rate 감소 시작 할 epoch 값
    parser.add_argument("--decay_epoch_num",
                        type=int,
                        help='set epoch num to start learning rate decay',
                        default=const_var_cyclegan.DECAY_EPOCH_NUM,
                        dest='decay_epoch_num')

    # replay buffer 사이즈
    parser.add_argument("--replay_buffer_max_size",
                        type=int,
                        help='set size of replay buffer',
                        default=const_var_cyclegan.REPLAY_BUFFER_MAX_SIZE,
                        dest='replay_buffer_max_size')

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
    from torch.utils.data import DataLoader

    from Common import const_var_cyclegan
    from CycleGAN import utils
    from CycleGAN.train import Trainer
    from CycleGAN.dataloader import FakeHome2RealHomeDataset
    from CycleGAN.model import GeneratorResNet, Discriminator
    from CycleGAN.loss import loss_fn_GAN, loss_fn_cycle, loss_fn_identity
    from CycleGAN.metric import GAN_loss, cycle_loss, identity_loss

    # GPU / CPU 설정
    device = const_var_cyclegan.DEVICE_CUDA if torch.cuda.is_available() else const_var_cyclegan.DEVICE_CPU

    # 모델 선언
    G_AB = GeneratorResNet()
    G_BA = GeneratorResNet()
    D_A = Discriminator()
    D_B = Discriminator()
    # 각 모델을 해당 디바이스로 이동
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)

    # optimizer 선언
    optimizerG = torch.optim.Adam(params=itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                  lr=args.learning_rate,
                                  betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(params=itertools.chain(D_A.parameters(), D_B.parameters()),
                                  lr=args.learning_rate,
                                  betas=(0.5, 0.999))

    # 학습용 데이터로더 선언
    train_dataloader = DataLoader(dataset=FakeHome2RealHomeDataset(data_dir=args.train_data_dir,
                                                                   mode_train_test=const_var_cyclegan.MODE_TRAIN),
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle)

    # 테스트용 데이터로더 선언
    test_dataloader = DataLoader(dataset=FakeHome2RealHomeDataset(data_dir=args.test_data_dir,
                                                                  mode_train_test=const_var_cyclegan.MODE_TEST))

    # replay buffer 선언
    replay_buffer_A = utils.ReplayBuffer(replay_buffer_max_size=args.replay_buffer_max_size)
    replay_buffer_B = utils.ReplayBuffer(replay_buffer_max_size=args.replay_buffer_max_size)

    # 모델 학습 객체 선언
    trainer = Trainer(G_AB=G_AB,
                      G_BA=G_BA,
                      D_A=D_A,
                      D_B=D_B,
                      optimizerG=optimizerG,
                      optimizerD=optimizerD,
                      loss_fn=None,
                      metric_fn=None,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      device=device,
                      replay_buffer_A=replay_buffer_A,
                      replay_buffer_B=replay_buffer_B)

    # 모델 학습
    trainer.running(num_epoch=args.num_epoch,
                    output_dir=args.output_dir,
                    decay_epoch_num=args.decay_epoch_num,
                    tracking_frequency=args.tracking_frequency,
                    checkpoint_file=args.checkpoint_file)


def main():

    # 경로 잡기
    set_path()

    # 인자 받기
    args = arguments()

    # 프로그램 실행
    run_program(args=args)


if __name__ == '__main__':
    main()
