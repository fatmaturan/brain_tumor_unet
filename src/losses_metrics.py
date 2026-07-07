import tensorflow as tf

smooth = 1e-6


def dice_coef(y_true, y_pred):
    """
    Dice Coefficient:
    Gerçek maske ile tahmin maskesinin örtüşme oranını ölçer.
    1'e yaklaştıkça daha iyi sonuç demektir.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """
    Dice Loss:
    Dice skorunun tersidir.
    Amaç dice değerini yükseltmektir.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def tversky_coef(y_true, y_pred, alpha=0.3, beta=0.7):
    """
    Tversky Coefficient:
    Dice'a benzer ama false positive ve false negative hatalarını ayrı ayrı cezalandırır.

    alpha: false positive cezası
    beta: false negative cezası

    beta yüksek olduğu için modelin tümörü kaçırmasını daha fazla cezalandırır.
    Bu da recall değerini artırmaya yardımcı olabilir.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    true_positive = tf.reduce_sum(y_true_f * y_pred_f)
    false_positive = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    false_negative = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

    return (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )


def tversky_loss(y_true, y_pred):
    """
    Tversky Loss:
    Tversky coefficient değerini yükseltmek için kullanılır.
    """
    return 1.0 - tversky_coef(y_true, y_pred, alpha=0.3, beta=0.7)


def bce_dice_loss(y_true, y_pred):
    """
    Eski kullandığın loss:
    Binary Crossentropy + Dice Loss
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def bce_tversky_loss(y_true, y_pred):
    """
    Yeni deneyeceğimiz loss:
    Binary Crossentropy + Tversky Loss

    Bu loss özellikle tümör gibi küçük alanların kaçırılmasını azaltmaya çalışır.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + tversky_loss(y_true, y_pred)


def iou(y_true, y_pred):
    """
    IoU:
    Kesişim / birleşim oranı.
    1'e yaklaştıkça daha iyi segmentation demektir.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def precision_m(y_true, y_pred):
    """
    Precision:
    Modelin tümör dediği alanların ne kadarının gerçekten tümör olduğunu ölçer.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    true_positive = tf.reduce_sum(y_true * y_pred)
    predicted_positive = tf.reduce_sum(y_pred)

    return (true_positive + smooth) / (predicted_positive + smooth)


def recall_m(y_true, y_pred):
    """
    Recall:
    Gerçek tümör alanının ne kadarını modelin yakaladığını ölçer.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    true_positive = tf.reduce_sum(y_true * y_pred)
    possible_positive = tf.reduce_sum(y_true)

    return (true_positive + smooth) / (possible_positive + smooth)