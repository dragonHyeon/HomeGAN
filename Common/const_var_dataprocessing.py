import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
IMAGE_FILE_NAME = '{0}_{1}.png'
ORIGINAL_JSON_DIR = '{0}/RES/original/json/'.format(PROJECT_ROOT_DIRECTORY)
LSGAN_DIR_TRAIN = '{0}/RES/processed/lsgan/train/'.format(PROJECT_ROOT_DIRECTORY)
LSGAN_DIR_TEST = '{0}/RES/processed/lsgan/test/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TRAIN_A = '{0}/RES/processed/cyclegan/train/a/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TRAIN_B = '{0}/RES/processed/cyclegan/train/b/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TEST_A = '{0}/RES/processed/cyclegan/test/a/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TEST_B = '{0}/RES/processed/cyclegan/test/b/'.format(PROJECT_ROOT_DIRECTORY)
SINGAN_DIR_TRAIN = '{0}/RES/processed/singan/train/'.format(PROJECT_ROOT_DIRECTORY)

# 원본 JSON 파일 키 값
KEY_PRE_VERTS = 'verts'
KEY_PRE_ID = 'id'
KEY_PRE_ROOM_NUM = 'room_num'
KEY_PRE_BBOX = 'bbox'
KEY_PRE_BBOX_MIN = 'min'
KEY_PRE_BBOX_MAX = 'max'
# 변환된 데이터에 대한 dict 키 값
KEY_POST_IMAGE = 'image'
KEY_POST_ID = 'id'
KEY_POST_ROOM_NUM = 'room_num'

# pre processing 시 사용되는 값
METER2PIXEL = 100
BORDER_PAD = 9
# post processing 시 사용되는 값
MAX_LENGTH = 2700
INITIAL_MAX_LENGTH = 0
# dataset save 시 사용되는 값
NUM_SINGAN_TRAIN = 1
PROPER_MAX_NUM_TEST = 200
