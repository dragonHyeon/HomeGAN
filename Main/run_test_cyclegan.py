import os, sys, argparse, time


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

    # 이미지 변환 방향
    direction_list = [
        const_var_cyclegan.A2B,
        const_var_cyclegan.B2A
    ]

    # parser 생성
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Test",
                                     description="* Run this to test the model.")

    # parser 인자 목록 생성
    # 이미지 변환 방향 선택
    parser.add_argument("--direction",
                        type=str,
                        help='select style transfer direction ({0} / {1})'.format(
                            const_var_cyclegan.A2B,
                            const_var_cyclegan.B2A
                        ),
                        choices=direction_list,
                        default=const_var_cyclegan.B2A,
                        dest="direction")

    # 입력 이미지 파일 디렉터리 위치
    parser.add_argument("--input_dir",
                        type=str,
                        help='set input image file directory',
                        default=const_var_cyclegan.INPUT_DIR,
                        dest='input_dir')

    # 불러올 체크포인트 파일 경로
    parser.add_argument("--checkpoint_file",
                        type=str,
                        help='set checkpoint file to load if exists',
                        default='../DATA/cyclegan/checkpoint/epoch00019.ckpt',
                        dest='checkpoint_file')

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=const_var_cyclegan.OUTPUT_DIR,
                        dest='output_dir')

    # 생성된 이미지 파일 저장될 폴더명
    parser.add_argument("--generated_folder_name",
                        type=str,
                        help='set generated folder name which generated images going to be saved',
                        default=time.time(),
                        dest="generated_folder_name")

    # parsing 한거 가져오기
    args = parser.parse_args()

    return args


def run_program(args):
    """
    * 테스트 실행
    :param args: 프로그램 실행 인자
    :return: None
    """

    import torch

    from Common import const_var_cyclegan
    from CycleGAN.test import Tester
    from CycleGAN.model import GeneratorResNet
    from CycleGAN import utils

    # GPU / CPU 설정
    device = const_var_cyclegan.DEVICE_CUDA if torch.cuda.is_available() else const_var_cyclegan.DEVICE_CPU

    # 체크포인트 파일 불러오기
    state = utils.load_checkpoint(filepath=args.checkpoint_file)

    # 모델 선언 및 가중치 불러오기
    G = GeneratorResNet()
    if args.direction == const_var_cyclegan.A2B:
        G.load_state_dict(state[const_var_cyclegan.KEY_STATE_G_AB])
    elif args.direction == const_var_cyclegan.B2A:
        G.load_state_dict(state[const_var_cyclegan.KEY_STATE_G_BA])
    # 모델을 해당 디바이스로 이동
    G.to(device)

    # 모델 테스트 객체 선언
    tester = Tester(G=G,
                    device=device)

    # 모델 테스트
    tester.running(input_dir=args.input_dir,
                   output_dir=args.output_dir,
                   generated_folder_name=args.generated_folder_name)


def main():

    # 경로 잡기
    set_path()

    # 인자 받기
    args = arguments()

    # 프로그램 실행
    run_program(args=args)


if __name__ == '__main__':
    main()
