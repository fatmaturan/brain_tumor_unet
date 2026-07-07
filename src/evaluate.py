import tensorflow as tf

from config import MODEL_PATH, BATCH_SIZE
from dataset import split_data, create_dataset
from losses_metrics import bce_dice_loss, dice_coef, iou, precision_m, recall_m


def main():
    x_train, x_val, x_test, y_train, y_val, y_test = split_data()

    test_dataset = create_dataset(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )

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

    results = model.evaluate(test_dataset)

    print("Test Loss:", results[0])
    print("Test Dice:", results[1])
    print("Test IoU:", results[2])
    print("Test Precision:", results[3])
    print("Test Recall:", results[4])


if __name__ == "__main__":
    main()