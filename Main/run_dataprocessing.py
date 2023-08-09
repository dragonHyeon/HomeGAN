import os, sys


def set_path():
    """
    * 경로 잡기
    """

    # 경로명 설정
    AppPath = os.path.dirname(os.path.abspath(os.getcwd()))
    SRC_DIR_NAME_Common = 'Common'
    SRC_DIR_NAME_DATA = 'DATA'
    SRC_DIR_NAME_DataProcessing = 'DataProcessing'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    DataProcessing = os.path.join(AppPath, SRC_DIR_NAME_DataProcessing)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)

    # 경로 추가
    AppPathList = [AppPath, Common, DATA, DataProcessing, Lib, LOG, Main, RES]
    for p in AppPathList:
        sys.path.append(p)


def run_program():
    """
    * 데이터 프로세싱 실행
    :return: None
    """

    from Common import const_var_dataprocessing
    from DataProcessing.data_processing import DatasetMaker

    # 데이터 프로세싱 객체 선언
    dataset_maker = DatasetMaker(json_dir=const_var_dataprocessing.ORIGINAL_JSON_DIR,
                                 lsgan_dir_train=const_var_dataprocessing.LSGAN_DIR_TRAIN,
                                 lsgan_dir_test=const_var_dataprocessing.LSGAN_DIR_TEST,
                                 cyclegan_dir_train_a=const_var_dataprocessing.CYCLEGAN_DIR_TRAIN_A,
                                 cyclegan_dir_test_a=const_var_dataprocessing.CYCLEGAN_DIR_TEST_A,
                                 singan_dir_train=const_var_dataprocessing.SINGAN_DIR_TRAIN)

    # 데이터 프로세싱 실행
    dataset_maker.running()


def main():

    # 경로 잡기
    set_path()

    # 프로그램 실행
    run_program()


if __name__ == '__main__':
    main()
