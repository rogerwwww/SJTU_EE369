#!python3
import numpy as np
from utility import *
import sys


class KNNDeco:
    def __init__(self, training_data, training_label, label_space, k):
        self.training_data = training_data
        self.training_label = training_label
        self.label_space = label_space
        self.k = k

    def __call__(self, test_data):
        ret = knn(self.training_data, self.training_label, test_data, self.label_space, self.k)
        return ret


def knn(training_data_set, training_label, testing_data, label_space, k=5):
    """
    k-Nearest-Neighbor method. 
    :param training_data_set: training data set, numpy array
    :param training_label: labels of training data, numpy array
    :param testing_data: single entity of testing data, numpy array
    :param label_space: full label space
    :param k: k-Nearest-Neighbor's k
    :return: label of result
    """
    distances = list(map(lambda x: np.linalg.norm(x - testing_data), training_data_set))
    vote = np.zeros(len(label_space))
    for i in range(k):
        min_index = np.argmin(distances)
        distances[min_index] = sys.maxsize
        vote[training_label[min_index]] += 1
    result = np.argmax(vote)
    return result

if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description='kNN method on MNIST and CIFAR data set.')
    parser.add_argument('-k', dest='k', nargs='?', default=3, type=int, help='k value of kNN, default=3')
    parser.add_argument('-d --data', dest='dataset', default='cifar', choices=['cifar', 'mnist'], nargs='?', type=str,
                        help='Indicate which data set to use (mnist or cifar), default=cifar')
    parser.add_argument('-r --result', dest='result', nargs='?', default='', type=str, help='test result filename')
    parser.add_argument('-j --jobs', dest='jobs', nargs='?', default=1, type=int, help='maximum parallel jobs allowed')

    args = parser.parse_args()

    print("Loading training and testing set...")
    if args.dataset == 'mnist':
        # Load all info
        labelSpace = list(range(10))
        trainingData, trainingLabel = unpack_mnist('Data/train-images.idx3-ubyte', 'Data/train-labels.idx1-ubyte')
        testingData, testingLabel = unpack_mnist('Data/t10k-images.idx3-ubyte', 'Data/t10k-labels.idx1-ubyte')

    else:
        labelSpace = CIFAR_LABEL_SPACE
        trainingData, trainingLabel = \
            unpack_cifar_feature('CIFAR-10/HOG+SVM classifier/data/features/train')
        testingData, testingLabel = \
            unpack_cifar_feature('CIFAR-10/HOG+SVM classifier/data/features/test')

    print("Training...")
    result = np.empty(testingData.shape[0])
    #i = 0
    try:
        with Pool(args.jobs) as pool:
            knn_iter = pool.imap(KNNDeco(trainingData, trainingLabel, labelSpace, args.k), testingData)
            result = list(tqdm(knn_iter, total=testingData.shape[0], ascii=True, ncols=75))
        '''
        for row in tqdm(testingData, ncols=75, ascii=True):
            result[i] = knn(trainingData, trainingLabel, row, labelSpace, k=args.k)
            tqdm.write('Classifying %d/%d' % (i + 1, testingData.shape[0]))
            tqdm.write('label = ' + str(labelSpace[testingLabel[i]]) + '\tresult = ' + str(labelSpace[int(result[i])]))
            i += 1
        '''
    except KeyboardInterrupt:
        # print result when program is terminated by user (ctrl-C).
        error = np.count_nonzero(result[:i] - testingLabel[:i])

        print('\nClassification terminated by user.')
        print('Training samples = %d' % trainingLabel.shape[0])
        print('Testing samples = %d' % i)
        print('Errors = %d' % error)
        print('Error rate = %2.4f%%' % (float(error) / i * 100))

    # print result when calculation complete
    error = np.count_nonzero(result - testingLabel)

    print_string = 'Classification completed.\n' \
                   'Training samples = %d\n' \
                   'Testing samples = %d\n' \
                   'Errors = %d\n' \
                   'Error rate = %2.4f%%' \
                   % (trainingLabel.shape[0], testingLabel.shape[0], error, error / testingLabel.shape[0] * 100)

    if args.result != '':
        with open(args.result, 'w') as test_result_file:
            test_result_file.write(print_string)

    print('\n', print_string)
