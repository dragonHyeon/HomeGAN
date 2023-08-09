import os
import shutil
from pathlib import Path


def make_parent_dir_if_not_exits(target_path):
    """
    * 파일의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    :param target_path: 파일명 포함 경로
    :return: None
    """

    # 경로 존재하지 않을 경우 생성
    if not os.path.exists(os.path.dirname(target_path)):
        # 경로 생성
        os.makedirs(os.path.dirname(target_path))


def copyfile_with_make_parent_dir(src, dst):
    """
    * 파일의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성하고 원본 파일 경로에서 대상 파일 경로로 파일 복사하기
    :param src: 복사할 원본 파일 경로
    :param dst: 복사할 대상 파일 경로
    :return: None
    """

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    make_parent_dir_if_not_exits(target_path=dst)

    # 원본 파일 경로에서 대상 파일 경로로 파일 복사
    shutil.copyfile(src=src, dst=dst)


def get_bottom_folder(path):
    """
    * 해당 경로의 가장 하위에 있는 폴더의 이름을 반환
    :param path: 경로
    :return: 가장 하위에 있는 폴더의 이름
    """

    return Path(path).parts[-1]


def get_second_bottom_folder(path):
    """
    * 해당 경로의 하위에서 두 번째에 있는 폴더의 이름을 반환
    :param path: 경로
    :return: 하위에서 두 번째에 있는 폴더의 이름
    """

    return Path(path).parts[-2]
