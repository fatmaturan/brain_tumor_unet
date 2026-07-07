import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from config import RAW_IMAGE_DIR, RAW_MASK_DIR, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, SEED


def get_image_mask_paths():
    image_files = sorted(os.listdir(RAW_IMAGE_DIR))

    image_paths = []
    mask_paths = []

    for file_name in image_files:
        image_path = os.path.join(RAW_IMAGE_DIR, file_name)
        mask_path = os.path.join(RAW_MASK_DIR, file_name)

        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)

    print(f"Toplam eşleşen image-mask sayısı: {len(image_paths)}")

    return image_paths, mask_paths


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype(np.float32) / 255.0

    return image


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Maske okunamadı: {path}")

    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return mask


def parse_data(image_path, mask_path):
    image, mask = tf.numpy_function(
        func=lambda x, y: (
            read_image(x.decode("utf-8")),
            read_mask(y.decode("utf-8"))
        ),
        inp=[image_path, mask_path],
        Tout=[tf.float32, tf.float32]
    )

    image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])

    return image, mask


def augment_data(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.08)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def create_dataset(image_paths, mask_paths, batch_size=2, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED)

    dataset = dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def split_data():
    image_paths, mask_paths = get_image_mask_paths()

    x_train, x_temp, y_train, y_temp = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.2,
        random_state=SEED
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
def augment_data(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.08)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask