import os, sys, argparse, time


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
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Test",
                                     description="* Run this to test the model.")

    # 생성할 이미지 개수
    parser.add_argument("--nums_to_generate",
                        type=int,
                        help='set number of images to generate',
                        default=const_var_singan.NUM_GENERATION,
                        dest='nums_to_generate')

    # 불러올 체크포인트 파일 경로
    parser.add_argument("--checkpoint_file",
                        type=str,
                        help='set checkpoint file to load if exists',
                        default='../DATA/singan/checkpoint/7/iter02000/checkpoint.ckpt',
                        dest='checkpoint_file')

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=const_var_singan.OUTPUT_DIR,
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

    from Common import const_var_singan
    from SinGAN import utils
    from SinGAN.test import Tester

    # GPU / CPU 설정
    device = const_var_singan.DEVICE_CUDA if torch.cuda.is_available() else const_var_singan.DEVICE_CPU

    # 체크포인트 파일 불러오기
    state = utils.load_checkpoint(filepath=args.checkpoint_file)

    # 모델 모음 불러오기
    Gs = state[const_var_singan.KEY_STATE_GS]
    # 각 모델을 해당 디바이스로 이동
    for G in Gs:
        G.to(device)

    # 시그마 모음 불러오기
    sigmas = state[const_var_singan.KEY_STATE_SIGMAS]

    # 이미지 피라미드 불러오기
    image_pyramids = state[const_var_singan.KEY_STATE_IMAGE_PYRAMIDS]

    # 모델 테스트 객체 선언
    tester = Tester(Gs=Gs,
                    sigmas=sigmas,
                    image_pyramids=image_pyramids,
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
