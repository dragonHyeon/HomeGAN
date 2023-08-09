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
    SRC_DIR_NAME_HomeGANSin = 'HomeGANSin'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'
    SRC_DIR_NAME_SinGAN = 'SinGAN'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    CycleGAN = os.path.join(AppPath, SRC_DIR_NAME_CycleGAN)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    HomeGANSin = os.path.join(AppPath, SRC_DIR_NAME_HomeGANSin)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)
    SinGAN = os.path.join(AppPath, SRC_DIR_NAME_SinGAN)

    # 경로 추가
    AppPathList = [AppPath, Common, CycleGAN, DATA, HomeGANSin, Lib, LOG, Main, RES, SinGAN]
    for p in AppPathList:
        sys.path.append(p)


def arguments():
    """
    * parser 이용하여 프로그램 실행 인자 받기
    :return: args
    """

    from Common import const_var_homegansin

    # parser 생성
    parser = argparse.ArgumentParser(prog="HomeGANSin",
                                     description="* Welcome to HomeGANSin!")

    # parser 인자 목록 생성
    # 생성할 이미지 개수
    parser.add_argument("--nums_to_generate",
                        type=int,
                        help='set number of floor plan images to generate',
                        default=const_var_homegansin.NUM_GENERATION,
                        dest='nums_to_generate')

    # 불러올 체크포인트 파일 경로 (SinGAN)
    parser.add_argument("--checkpoint_file_singan",
                        type=str,
                        help='set checkpoint file to load if exists (SinGAN)',
                        default='../DATA/singan/checkpoint/1/iter02000/checkpoint.ckpt',
                        dest='checkpoint_file_singan')

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
                        default=const_var_homegansin.OUTPUT_DIR,
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

    from Common import const_var_homegansin
    from HomeGANSin.test import Tester
    from HomeGANSin import utils
    from CycleGAN.model import GeneratorResNet

    # GPU / CPU 설정
    device = const_var_homegansin.DEVICE_CUDA if torch.cuda.is_available() else const_var_homegansin.DEVICE_CPU

    # 체크포인트 파일 불러오기
    state_singan = utils.load_checkpoint(filepath=args.checkpoint_file_singan)
    state_cyclegan = utils.load_checkpoint(filepath=args.checkpoint_file_cyclegan)

    # 모델 모음 불러오기 (SinGAN)
    singan = state_singan[const_var_homegansin.KEY_STATE_GS]
    # 각 모델을 해당 디바이스로 이동
    for G in singan:
        G.to(device)

    # 시그마 모음 불러오기 (SinGAN)
    sigmas = state_singan[const_var_homegansin.KEY_STATE_SIGMAS]

    # 이미지 피라미드 불러오기 (SinGAN)
    image_pyramids = state_singan[const_var_homegansin.KEY_STATE_IMAGE_PYRAMIDS]

    # 모델 선언 및 가중치 불러오기 (CycleGAN)
    cyclegan = GeneratorResNet()
    cyclegan.load_state_dict(state_cyclegan[const_var_homegansin.KEY_STATE_G_BA])
    # 모델을 해당 디바이스로 이동
    cyclegan.to(device)

    # 모델 테스트 객체 선언
    tester = Tester(singan=singan,
                    sigmas=sigmas,
                    image_pyramids=image_pyramids,
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
