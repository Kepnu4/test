import os
import math
import argparse
import random
import numpy as np
import tensorflow as tf

import utils

def get_train_val_data(args):
    images = np.load(args.data_path)
    np.random.shuffle(images)
    n_val = int(images.shape[0] * args.val_ratio)
    n_train = images.shape[0] - n_val
    train_images, val_images = images[:n_train], images[n_train:]

    random_state = np.random.RandomState(args.seed)
    train_labels = random_state.uniform(args.rotate_angle_low, args.rotate_angle_high, n_train) 
    train_labels = train_labels.astype(np.float32)
    val_labels = random_state.uniform(args.rotate_angle_low, args.rotate_angle_high, n_val) 
    val_labels = vaL_labels.astype(np.float32)

    train_dataset, train_feed_dict = utils.get_dataset_from_np(train_images, train_labels, is_train=True, **args)
    val_dataset, val_feed_dict = utils.get_dataset_from_np(val_images, val_labels, is_train=False, **args)
    
    return train_dataset, train_feed_dict, val_dataset, val_feed_dict

def get_model(args):
    inputs = tf.keras.Input(shape=(args.image_height, args.image_width, args.image_channels))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', 
                               activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', 
                               activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

class InitIteratorsCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterators, feed_dicts):
            super().__init__()
            self.iterators = iterators
            self.feed_dicts = feed_dicts

    def on_epoch_begin(self, epoch, logs=None):
        sess = tf.keras.backend.get_session()
        for iterator, feed_dict in zip(self.iterators, self.feed_dicts):
            sess.run(iterator.initializer, feed_dict=feed_dict)


def main(args):
    train_data, train_feed_dict, val_data, val_feed_dict = get_train_val_data(args)
    train_iterator = train_data.make_initializable_iterator()
    val_iterator = val_data.make_initializable_iterator()

    model = get_model(args)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=angle_loss)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(args.model_save_path, save_best_only=True)
    model.fit(train_iterator, epochs=args.epochs, steps_per_epoch=1000, validation_data=val_iterator,
              validation_steps=200,
              callbacks=[InitIteratorsCallback([train_iterator, val_iterator], 
                                               [train_feed_dict, val_feed_dict]),
                         model_checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train_data.npy')
    parser.add_argument('--model_save_path', type=str, default='models/model.h5')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='train/val ratio')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_channels', type=int, default=1)
    parser.add_argument('--rotate_angle_low', type=float, default=0.0, help='min rotate angle div by pi')
    parser.add_argument('--rotate_angle_high', type=float, default=1.0, help='max rotate angle div by pi')
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)
