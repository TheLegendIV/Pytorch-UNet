DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_LOAD = False
DEFAULT_SCALE = 0.5
DEFAULT_AMP = False
DEFAULT_BILINEAR = False
DEFAULT_CLASSES = 4
DEFAULT_CLASS_WEIGHTS = None
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_PREDICT_SCALE = 0.5
DEFAULT_SAVE_CHECKPOINT = False

## ARCADE dataset
dir_train_imgs_src = r'./arcade/data/syntax/train/images/'
dir_train_annotations = r'./arcade/data/syntax/train/annotations/train.json'

dir_val_imgs_src = r'./arcade/data/syntax/val/images/'
dir_val_annotations = r'./arcade/data/syntax/val/annotations/val.json'

dir_img = r'./arcade/imgs/train/'
dir_val = r'./arcade/imgs/val/'
dir_train_mask = r'./arcade/masks/train/'
dir_val_mask = r'./arcade/masks/val/'
dir_preview_colorized = r'./arcade/preview/colorized/'
dir_preview_raw = r'./arcade/preview/raw/'
dir_checkpoint = r'./checkpoints/'

groupings = "6,7,8,9,10,11,12;1,2,3,4,20,21,22,23;13,14,15,24,16,17,18,25,19" # LAD; RCA; LCX;