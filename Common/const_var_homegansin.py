import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
OUTPUT_DIR = '{0}/DATA/homegansin/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR_SUFFIX_CHECKPOINT = 'checkpoint'
OUTPUT_DIR_SUFFIX_PICS = 'pics'
OUTPUT_DIR_SUFFIX_GENERATED_IMG = 'generated_img/{0}'
CHECKPOINT_FILE_NAME = 'epoch{:05d}.ckpt'
PICS_FILE_NAME = 'epoch{:05d}.png'
CHECKPOINT_BEST_FILE_NAME = 'best_model.ckpt'
GENERATED_IMG_FILE_NAME = '{0}.png'

# 학습 / 테스트 모드
MODE_TRAIN = 'train'
MODE_TEST = 'test'

# 디바이스 종류
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

# 판별자 참, 거짓 정보
REAL_LABEL = 1
FAKE_LABEL = 0

# 하이퍼 파라미터
LEARNING_RATE = 0.0002
BATCH_SIZE = 4
NUM_EPOCH = 30
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]

# 옵션 값
SHUFFLE = True
TRACKING_FREQUENCY = 1
NUM_PICS_LIST = 10
NUM_GENERATION = 17
PROPER_MAX_NUM_TEST = 200

# 그 외 기본 설정 값
RESIZE_SIZE = 64

# state 저장시 딕셔너리 키 값
KEY_STATE_GS = 'Gs'
KEY_STATE_SIGMAS = 'sigmas'
KEY_STATE_IMAGE_PYRAMIDS = 'image_pyramids'
KEY_IMGPYR_IMAGE = 'image'
KEY_STATE_G_BA = 'G_BA'

# 초기 값
INITIAL_START_EPOCH_NUM = 1
INITIAL_BEST_BCE_LOSS = 100000
