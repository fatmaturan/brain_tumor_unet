import os
import random
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, PREDICTION_DIR, THRESHOLD
from dataset import split_data, read_image, read_mask
from losses_metrics import bce_dice_loss, dice_coef, iou, precision_m, recall_m
from utils import create_dir, save_prediction_plot


def main():
    create_dir(PREDICTION_DIR)

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

    sample_count = min(10, len(x_test))
    selected_indices = random.sample(range(len(x_test)), sample_count)

    for i, idx in enumerate(selected_indices):
        image = read_image(x_test[idx])
        true_mask = read_mask(y_test[idx])

        input_image = np.expand_dims(image, axis=0)

        pred_mask = model.predict(input_image, verbose=0)[0]
        pred_mask = (pred_mask > THRESHOLD).astype(np.float32)

        save_prediction_plot(
            image=image,
            true_mask=true_mask,
            pred_mask=pred_mask,
            save_path=os.path.join(PREDICTION_DIR, f"prediction_{i+1}.png")
        )

    print("Tahmin görselleri kaydedildi.")
    print("Kullanılan threshold:", THRESHOLD)


if __name__ == "__main__":
    main()