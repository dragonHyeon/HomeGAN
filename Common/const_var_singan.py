import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
DATA_PATH_TRAIN = '{0}/RES/processed/singan/train/000d0395709d2a16e195c6f0189155c4_6.png'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR = '{0}/DATA/singan/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR_SUFFIX_CHECKPOINT = 'checkpoint/{0}'
OUTPUT_DIR_SUFFIX_CHECKPOINT_ITER ='iter{:05d}'
OUTPUT_DIR_SUFFIX_GENERATED_IMG = 'generated_img/{0}'
CHECKPOINT_FILE_NAME = 'checkpoint.ckpt'
CHECKPOINT_TRAIN_IMG_FILE_NAME = 'trained_image.jpg'
CHECKPOINT_RESULT_IMG_FILE_NAME = 'scale{0}_{1}.png'
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
LEARNING_RATE = 5e-4
BATCH_SIZE = 1
NUM_ITER = 2000
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
DECAY_ITER_NUM = 50
REPLAY_BUFFER_MAX_SIZE = 50

# 옵션 값
SHUFFLE = True
TRACKING_FREQUENCY = 1
NUM_PICS_LIST = 10
NUM_GENERATION = 10

# 그 외 기본 설정 값
RESIZE_SIZE = 256
LAMBDA_REC = 10.0

# state 저장시 딕셔너리 키 값
KEY_STATE_GS = 'Gs'
KEY_STATE_DS = 'Ds'
KEY_STATE_SIGMAS = 'sigmas'
KEY_STATE_IMAGE_PYRAMIDS = 'image_pyramids'

# score 저장시 딕셔너리 키 값
KEY_SCORE_G = 'scoreG'
KEY_SCORE_D = 'scoreD'

# image pyramids 저장시 딕셔너리 키 값
KEY_IMGPYR_IMAGE = 'image'
KEY_IMGPYR_MEAN = 'mean'
KEY_IMGPYR_STD = 'std'

# 초기 값
INITIAL_START_ITER_NUM = 1
INITIAL_BEST_BCE_LOSS = 100000
