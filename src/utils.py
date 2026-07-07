import os
import matplotlib.pyplot as plt


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_history(history, save_path):
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history["dice_coef"], label="Train Dice")
    plt.plot(history.history["val_dice_coef"], label="Val Dice")
    plt.title("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history.history["iou"], label="Train IoU")
    plt.plot(history.history["val_iou"], label="Val IoU")
    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    if "precision_m" in history.history and "recall_m" in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history["precision_m"], label="Train Precision")
        plt.plot(history.history["val_precision_m"], label="Val Precision")
        plt.plot(history.history["recall_m"], label="Train Recall")
        plt.plot(history.history["val_recall_m"], label="Val Recall")
        plt.title("Precision / Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_prediction_plot(image, true_mask, pred_mask, save_path):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("MRI Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap="gray")
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()