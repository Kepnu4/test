import tensorflow as tf

def get_dataset_from_np(images, labels, args, is_train=True):
    images_pl = tf.placeholder(images.dtype, images.shape)
    labels_pl = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((images_pl, labels_pl))

    dataset = dataset.map(lambda x, y: prepare_image(x, y, args, is_train=True))
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)

    feed_dict = { images_pl: images, labels_pl: labels }
    return dataset, feed_dict

def prepare_image(image, angle, args, is_train=False):
    if is_train:
        angle = tf.random.uniform((1,), args.rotate_angle_low, args.rotate_angle_high) 
    image = tf.cast(image, tf.float32)
    image = tf.contrib.image.rotate(image, angle * math.pi * 2, interpolation='BILINEAR')
    image = image / tf.constant(256, tf.float32)
    return image, angle

def angle_loss(y_true, y_pred):
    angle_1 = tf.math.abs(y_true - y_pred)
    angle_2 = tf.constant(1, tf.float32) - angle_1
    return tf.reduce_mean(tf.square(tf.math.minimum(angle_1, angle_2)))
