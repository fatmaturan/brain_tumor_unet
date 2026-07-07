import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model


def conv_block(inputs, num_filters, dropout_rate=0.0):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


def encoder_block(inputs, num_filters, dropout_rate=0.0):
    x = conv_block(inputs, num_filters, dropout_rate)
    p = MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters, dropout_rate=0.0):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate)
    return x


def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32, 0.05)
    s2, p2 = encoder_block(p1, 64, 0.05)
    s3, p3 = encoder_block(p2, 128, 0.10)
    s4, p4 = encoder_block(p3, 256, 0.15)

    b1 = conv_block(p4, 512, 0.25)

    d1 = decoder_block(b1, s4, 256, 0.15)
    d2 = decoder_block(d1, s3, 128, 0.10)
    d3 = decoder_block(d2, s2, 64, 0.05)
    d4 = decoder_block(d3, s1, 32, 0.05)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return Model(inputs, outputs, name="Improved_U-Net")