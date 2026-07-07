import numpy as np
import tensorflow as tf

from config import MODEL_PATH
from dataset import split_data, read_image, read_mask
from losses_metrics import bce_dice_loss, dice_coef, iou, precision_m, recall_m


SMOOTH = 1e-6


def calculate_dice(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    intersection = np.sum(y_true * y_pred)

    return (2.0 * intersection + SMOOTH) / (
        np.sum(y_true) + np.sum(y_pred) + SMOOTH
    )


def calculate_iou(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return (intersection + SMOOTH) / (union + SMOOTH)


def main():
    x_train, x_val, x_test, y_train, y_val, y_test = split_data()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coef": dice_coef,
            "iou": iou,
            "precision_m": precision_m,
            "recall_m": recall_m
        }
    )

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("Threshold denemesi başlıyor...")
    print("-" * 50)

    best_threshold = 0.5
    best_dice = 0.0
    best_iou = 0.0

    for threshold in thresholds:
        dice_scores = []
        iou_scores = []

        for image_path, mask_path in zip(x_test, y_test):
            image = read_image(image_path)
            true_mask = read_mask(mask_path)

            input_image = np.expand_dims(image, axis=0)
            pred_mask = model.predict(input_image, verbose=0)[0]

            pred_mask = (pred_mask > threshold).astype(np.float32)

            dice_scores.append(calculate_dice(true_mask, pred_mask))
            iou_scores.append(calculate_iou(true_mask, pred_mask))

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)

        print(f"Threshold: {threshold:.2f} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_iou = mean_iou
            best_threshold = threshold

    print("-" * 50)
    print(f"En iyi threshold: {best_threshold}")
    print(f"En iyi Dice: {best_dice:.4f}")
    print(f"En iyi IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()