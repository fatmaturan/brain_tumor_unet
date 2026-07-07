import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_MASK_DIR = os.path.join(BASE_DIR, "data", "raw", "masks")

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet_model_gpu_improved.h5")
PREDICTION_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

THRESHOLD = 0.30