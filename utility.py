import time
import struct
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.externals import joblib
from tqdm import tqdm

CIFAR_LABEL_SPACE = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class UniversalTimer:
    def __init__(self):
        self.startTime = time.time()

    def current_time(self):
        return time.time() - self.startTime


def print_run_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        temp = func(*args, **kwargs)
        finish_time = time.time()
        print('-> %s has been running for %f seconds.' % (func.__name__, (finish_time - start_time)))
        return temp
    return wrapper


class InPoolDeco(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            ret = self.func(*args, **kwargs)
        except KeyboardInterrupt:
            return
        return ret


def unpack_mnist(data_set_addr, label_addr):
    """
    Unpack data from MNIST handwriting data set.
    :param data_set_addr: data set file address
    :param label_addr: label file address
    :return: two numpy arrays containing data set and labels
    """
    # Load data set
    bin_file = open(data_set_addr, 'rb')
    buf = bin_file.read()

    index = 0
    magic_num, num_images, num_rows, num_columns = struct.unpack_from('>IIII', buf, index)
    data_set = np.empty((num_images, 784))

    for i in tqdm(range(num_images), ncols=75, ascii=True):
        im = struct.unpack_from('>784B', buf, struct.calcsize('>IIII') + i * struct.calcsize('>784B'))
        im = np.array(im)
        data_set[i, :] = im

    # Load labels
    bin_file = open(label_addr, 'rb')
    buf = bin_file.read()

    index = 0
    magic_num, num_images = struct.unpack_from('>II', buf, index)
    unpack_str = '>'
    unpack_str += str(num_images)
    unpack_str += 'B'
    labels = struct.unpack_from(unpack_str, buf, struct.calcsize('>II'))
    labels = np.array(labels)

    return data_set, labels


def unpack_cifar_feature(folder_path, max_num=None):
    """
    Unpack cifar feature
    :param folder_path: target folder path
    :param label_space: optional input label space
    :param max_num: optional maximum number
    :return: 
    """
    data_set = []
    label_set = []
    file_list = os.listdir(folder_path)
    if max_num:
        file_list = file_list[:: int(len(file_list)/max_num)]
    for single_file in tqdm(file_list, ncols=75, ascii=True):
        full_path = os.path.join(folder_path, single_file)
        with open(full_path, 'rb') as fo:
            data = joblib.load(fo)
            data_set.append(data[: 324])
            label_set.append(int(data[324]))
    return np.array(data_set), np.array(label_set)


def sort_data_by_label(data_set, label_set):
    label_sort_result = np.lexsort((label_set,))
    data_set = data_set[label_sort_result]
    label_set = label_set[label_sort_result]
    return data_set, label_set


def threshold(img, thres=127):
    """
    Convert an image into a binary image.
    :param img: matrix of the original image
    :param thres: threshold value
    :return: converted binary image
    """
    img = np.array(img)
    img = ((img > thres) * 1).astype('uint8')
    return img


def show_img_vector(img):
    im = np.array(img)
    im = im.reshape(28, 28)

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.show()


def write_json(filename, data):
    with open(filename, 'w+') as json_file:
        json_file.write(json.dumps(data))


def read_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)
