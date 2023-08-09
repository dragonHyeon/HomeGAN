import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
LSGAN_DIR_TRAIN = '{0}/RES/processed/lsgan/train/'.format(PROJECT_ROOT_DIRECTORY)
LSGAN_DIR_TEST = '{0}/RES/processed/lsgan/test/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TRAIN_A = '{0}/RES/processed/cyclegan/train/a/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TRAIN_B = '{0}/RES/processed/cyclegan/train/b/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TEST_A = '{0}/RES/processed/cyclegan/test/a/'.format(PROJECT_ROOT_DIRECTORY)
CYCLEGAN_DIR_TEST_B = '{0}/RES/processed/cyclegan/test/b/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR = '{0}/DATA/lsgan/'.format(PROJECT_ROOT_DIRECTORY)
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
NUM_GENERATION = 2000
PROPER_MAX_NUM_TEST = 200

# 그 외 기본 설정 값
RESIZE_SIZE = 64

# state 저장시 딕셔너리 키 값
KEY_STATE_MODEL_G = 'modelG'
KEY_STATE_MODEL_D = 'modelD'
KEY_STATE_OPTIMIZER_G = 'optimizerG'
KEY_STATE_OPTIMIZER_D = 'optimizerD'
KEY_STATE_EPOCH = 'epoch'

# score 저장시 딕셔너리 키 값
KEY_SCORE_G = 'scoreG'
KEY_SCORE_D = 'scoreD'

# 초기 값
INITIAL_START_EPOCH_NUM = 1
INITIAL_BEST_BCE_LOSS = 100000
