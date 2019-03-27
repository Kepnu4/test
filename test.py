import argparse
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2 as cv

import utils

def angle_dist(y_pred, y_true):
    angle_1 = np.abs(y_pred - y_true)
    angle_2 = 1.0 - angle_1
    return np.minimum(angle_1, angle_2)
    

def rotate_image(image, angle):
    center = (image.shape[0] / 2.0, image.shape[1] / 2.0)
    rot_mat = cv.getRotationMatrix2D(center, angle * 360, 1.0)
    result = cv.warpAffine(image, rot_mat, (image.shape[0], image.shape[1]),
                           flags=cv.INTER_CUBIC)
    return np.expand_dims(result, -1)
    

def main(args):
    model = tf.keras.models.load_model(args.model_path, custom_objects={'angle_loss': utils.angle_loss})
    data_arr = np.load(args.images_path)
    
    errors = []
    examples = []
    for data in data_arr:
        images, cl = data['images'], data['class']
        images = np.expand_dims(np.reshape(np.array(images), (-1, 28, 28)), -1)
        angles = np.random.uniform(0, 1, (images.shape[0], 1))

        rotated_images = [rotate_image(image, angle) for image, angle in zip(images, angles)] 
        rotated_images = np.array(rotated_images) / 256.0
        predictions = model.predict(rotated_images)
        errors = angle_dist(predictions, angles)
        examples.append((rotated_images[:5], angles[:5], predictions[:5]))


    errors = errors * 360
    print('mean error: {:.3f} degrees'.format(np.mean(errors)))
    examples_image = None
    for images, angles, preds in examples:
        resized_images = []
        for i in range(images.shape[0]):
            image = cv.resize(images[i], (256, 256), interpolation=cv.INTER_CUBIC) * 256 
            cv.putText(image, 'truth = {:.2f}'.format(angles[i][0] * 360), (10, 20), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       0.7, (255))
            cv.putText(image, 'prediction = {:.2f}'.format(preds[i][0] * 360), (10, 40), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       0.7, (255))
            resized_images.append(image)
        resized_images = np.array(resized_images)
            
        if examples_image is None:
            examples_image = np.concatenate(resized_images, axis=1)
        else:
            examples_image = np.concatenate((examples_image, np.concatenate(resized_images, axis=1)), axis=0)
    cv.imwrite('images/examples.png', examples_image)

    plt.plot(sorted(errors), [i / len(errors) * 100.0 for i in range(len(errors))])
    plt.xlabel('error')
    plt.ylabel('percent')
    plt.savefig('images/plot.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/model.h5')
    parser.add_argument('--images_path', type=str, default='data/test_data.npy')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)
