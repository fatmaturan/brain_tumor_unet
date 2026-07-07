import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from dataset import split_data, read_image, read_mask
from losses_metrics import (
    dice_coef,
    iou,
    precision_m,
    recall_m,
    bce_dice_loss
)

MODEL_PATH = "../models/best_unet_model_gpu_improved.h5"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "iou": iou,
        "precision_m": precision_m,
        "recall_m": recall_m,
        "bce_dice_loss": bce_dice_loss
    }
)

print("Model yüklendi.")

# Test verilerini al
_, _, x_test, _, _, y_test = split_data()

y_true_all = []
y_pred_all = []

for img_path, mask_path in zip(x_test, y_test):

    image = read_image(img_path)
    mask = read_mask(mask_path)

    pred = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )[0]

    pred = (pred > 0.5).astype(np.uint8)

    y_true_all.extend(mask.flatten())
    y_pred_all.extend(pred.flatten())

cm = confusion_matrix(
    y_true_all,
    y_pred_all
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Background", "Tumor"]
)

disp.plot(cmap="Blues")

plt.title("Brain Tumor Confusion Matrix")

plt.savefig("../outputs/brain_confusion_matrix.png")

plt.show()

print("Kaydedildi:")
print("../outputs/brain_confusion_matrix.png")