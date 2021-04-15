from yacs.config import CfgNode as CN


# Create root config node
_C = CN()
# Config name
_C.NAME = "resnet18_base"
# Config version to manage version of configuration names and default
_C.VERSION = "0.1"


# ----------------------------------------
# System config
# ----------------------------------------
_C.SYSTEM = CN()

# Number of workers for dataloader
_C.SYSTEM.NUM_WORKERS = 8
# Use GPU for training and inference. Default is True
_C.SYSTEM.GPU = True
# Random seed for seeding everything (NumPy, Torch,...)
_C.SYSTEM.SEED = 0
# Use half floating point precision
_C.SYSTEM.FP16 = True
# FP16 Optimization level. See more at: https://nvidia.github.io/apex/amp.html#opt-levels
_C.SYSTEM.OPT_L = "O2"


# ----------------------------------------
# Directory name config
# ----------------------------------------
_C.DIRS = CN()

# Train, Validation and Testing image folders
_C.DIRS.DATA = './Split_data'
# Trained weights folder
_C.DIRS.WEIGHTS = "./weights/"
# Inference output folder
_C.DIRS.OUTPUTS = "./outputs/"
# Training log folder
_C.DIRS.LOGS = "./logs/"


# ----------------------------------------
# Datasets config
# ----------------------------------------
_C.DATA = CN()

# Create small subset to debug
_C.DATA.DEBUG = False
# Image input channel for training
_C.DATA.INP_CHANNEL = 3
# Image Width and Height for training
_C.DATA.SIZE = (512,512)
_C.DATA.NORMALIZE = False
# For CSV loading dataset style
# If dataset is contructed as folders with one class for each folder, see ImageFolder dataset style
# Train, Validation and Test CSV files
_C.DATA.CSV = CN()
_C.DATA.CSV.TRAIN = "Split_data/data/train.csv"
_C.DATA.CSV.VALID = "Split_data/data/valid.csv"
_C.DATA.CSV.TEST = "Split_data/data/test.csv"
# Fixed Resized Crop all images
_C.DATA.CROP_RESIZE = CN({"ENABLED": True})
_C.DATA.CROP_RESIZE.NEW_SIZE = 64
# Dataset augmentations style (albumentations / randaug / augmix)
_C.DATA.AUGMENT = ""
# For randaug augmentation. For augmix or albumentations augmentation, refer to those other section
_C.DATA.RANDAUG = CN()
# Number of augmentations picked for each iterations. Default is 2
_C.DATA.RANDAUG.N = 2
# Amptitude of augmentation transform (0 < M < 30). Default is 27
_C.DATA.RANDAUG.M = 27
# Use ranged amptitude for augmentations transforms. Default is False.
_C.DATA.RANDAUG.RANDOM_MAGNITUDE = False

# ----------------------------------------
# Training config
# ----------------------------------------
_C.TRAIN = CN()

# Number of training cycles
_C.TRAIN.NUM_CYCLES = 1
# Number of epoches for each cycle. Length of epoches list must equals number of cycle
_C.TRAIN.EPOCHES = 100
# Training batchsize
_C.TRAIN.BATCH_SIZE = 200



# ----------------------------------------
# Solver config
# ----------------------------------------
_C.SOLVER = CN()

# Solver algorithm
_C.SOLVER.OPTIMIZER = "adam"
# Solver scheduler (constant / step / cyclical)
_C.SOLVER.SCHEDULER = "cyclical"
# Warmup length. Set 0 if do not want to use
_C.SOLVER.WARMUP_LENGTH = 0
# Use gradient accumulation. If not used, step equals 1
_C.SOLVER.GD_STEPS = 1
# Starting learning rate (after warmup, if used)
_C.SOLVER.BASE_LR = 1e-2
# Weight decay coeffs
_C.SOLVER.WEIGHT_DECAY = 1e-2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
# Stochastic weights averaging
_C.SOLVER.SWA = CN({"ENABLED": False})
# SWA starting epoch
_C.SOLVER.SWA.START_EPOCH = 10
# SWA update frequency (iterations)
_C.SOLVER.SWA.FREQ = 5
# SWA decay coeff for moving average
_C.SOLVER.SWA.DECAY = 0.999


# ----------------------------------------
# Loss function config
# ----------------------------------------
_C.LOSS = CN()

# Loss function (ce / focal / dice)
_C.LOSS.NAME = "ce"
_C.LOSS.WEIGHTED_LOSS = False

# ----------------------------------------
# Model config
# ----------------------------------------
_C.MODEL = CN()

# Classification model arch
# _C.MODEL.NAME = "efficientnet_b0"
_C.MODEL.NAME = "resnet18"
# Load ImageNet pretrained weights
_C.MODEL.PRETRAINED = True
# Classification head
_C.MODEL.CLS_HEAD = 'linear'
# Number of classification class
_C.MODEL.NUM_CLASSES = 3

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`