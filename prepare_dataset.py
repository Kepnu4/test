import os
import random
import argparse
import numpy as np

def main(args):
    train_data = []
    test_data = []

    classes_data = []
    for cl in args.classes:
        classes_data.append((np.load(os.path.join(args.root_dir, cl + '.npy')), cl))

    total_train_images = 0
    total_test_images = 0
    total_images = {}
    for data, cl in classes_data:
        random.shuffle(data)
        n_test = int(len(data) * args.test_ratio)
        n_train = len(data) - n_test
        train_data.extend(data[:n_train])
        total_train_images += n_train
        test_data.append({ 'images': data[n_train:n_train + n_test], 'class': cl})
        total_test_images += n_test
        total_images[cl] = (n_train, n_test)

    train_data = np.array(train_data)
    train_data = np.expand_dims(np.reshape(train_data, (train_data.shape[0], 28, 28)), -1)
    np.save(os.path.join(args.root_dir, 'train_data.npy'), np.array(train_data))
    np.save(os.path.join(args.root_dir, 'test_data.npy'), np.array(test_data))

    print('total train images: {}, total test images: {}'.format(total_train_images, total_test_images))
    print('number of images by class:')
    for cl, n in total_images.items():
        print('{}: train {}, test {}'.format(cl, n[0], n[1]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--classes', type=str, default='cat,tiger,zebra', help='list of classes')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='train/test ratio')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.classes = args.classes.split(',')

    random.seed(args.seed)

    main(args)
