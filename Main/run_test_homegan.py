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
    SRC_DIR_NAME_HomeGAN = 'HomeGAN'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_LSGAN = 'LSGAN'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    CycleGAN = os.path.join(AppPath, SRC_DIR_NAME_CycleGAN)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    HomeGAN = os.path.join(AppPath, SRC_DIR_NAME_HomeGAN)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    LSGAN = os.path.join(AppPath, SRC_DIR_NAME_LSGAN)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)

    # 경로 추가
    AppPathList = [AppPath, Common, CycleGAN, DATA, HomeGAN, Lib, LOG, LSGAN, Main, RES]
    for p in AppPathList:
        sys.path.append(p)


def arguments():
    """
    * parser 이용하여 프로그램 실행 인자 받기
    :return: args
    """

    from Common import const_var_homegan

    # parser 생성
    parser = argparse.ArgumentParser(prog="HomeGAN",
                                     description="* Welcome to HomeGAN!")

    # parser 인자 목록 생성
    # 생성할 이미지 개수
    parser.add_argument("--nums_to_generate",
                        type=int,
                        help='set number of floor plan images to generate',
                        default=const_var_homegan.NUM_GENERATION,
                        dest='nums_to_generate')

    # 불러올 체크포인트 파일 경로 (LSGAN)
    parser.add_argument("--checkpoint_file_lsgan",
                        type=str,
                        help='set checkpoint file to load if exists (LSGAN)',
                        default='../DATA/lsgan/checkpoint/epoch00003.ckpt',
                        dest='checkpoint_file_lsgan')

    # 불러올 체크포인트 파일 경로 (CycleGAN)
    parser.add_argument("--checkpoint_file_cyclegan",
                        type=str,
                        help='set checkpoint file to load if exists (CycleGAN)',
                        default='../DATA/cyclegan/checkpoint/epoch00019.ckpt',
                        dest='checkpoint_file_cyclegan')

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=const_var_homegan.OUTPUT_DIR,
                        dest='output_dir')

    # 생성된 이미지 파일 저장될 폴더명
    parser.add_argument("--generated_folder_name",
                        type=str,
                        help='set generated folder name which generated floor plan images going to be saved',
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

    from Common import const_var_homegan
    from HomeGAN.test import Tester
    from LSGAN.model import Generator
    from CycleGAN.model import GeneratorResNet
    from HomeGAN import utils

    # GPU / CPU 설정
    device = const_var_homegan.DEVICE_CUDA if torch.cuda.is_available() else const_var_homegan.DEVICE_CPU

    # 체크포인트 파일 불러오기
    state_lsgan = utils.load_checkpoint(filepath=args.checkpoint_file_lsgan)
    state_cyclegan = utils.load_checkpoint(filepath=args.checkpoint_file_cyclegan)

    # 모델 선언 및 가중치 불러오기 (LSGAN)
    lsgan = Generator()
    lsgan.load_state_dict(state_lsgan[const_var_homegan.KEY_STATE_MODEL_G])
    # 모델을 해당 디바이스로 이동
    lsgan.to(device)

    # 모델 선언 및 가중치 불러오기 (CycleGAN)
    cyclegan = GeneratorResNet()
    cyclegan.load_state_dict(state_cyclegan[const_var_homegan.KEY_STATE_G_BA])
    # 모델을 해당 디바이스로 이동
    cyclegan.to(device)

    # 모델 테스트 객체 선언
    tester = Tester(lsgan=lsgan,
                    cyclegan=cyclegan,
                    device=device)

    # 모델 테스트
    tester.running(nums_to_generate=args.nums_to_generate,
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
