#!python3
from utility import *
import numpy as np


class FFDeco:
    def __init__(self, v, w, gamma, theta):
        self.v = v
        self.w = w
        self.gamma = gamma
        self.theta = theta

    def __call__(self, x):
        return feed_forward(x, self.v, self.w, self.gamma, self.theta)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feed_forward(x, v, w, gamma, theta):
    """
    feed forward function in BP
    :param x: input vector
    :param v: 1st transfer matrix
    :param w: 2nd transfer matrix
    :param gamma: 1st threshold
    :param theta: 2nd threshold
    :return: b, y_hat
    """
    alpha = np.dot(v, x)
    b = sigmoid(alpha - gamma)
    beta = np.dot(w, b)
    y_hat = sigmoid(beta - theta)
    return b, y_hat


def standard_back_propagation(eta, x, y, b, y_hat, w):
    """
    Standard BP routine
    :param eta: learning rate
    :param x: input vector
    :param y: training result vector
    :param b: output of hidden layer
    :param y_hat: estimated result (output of output layer)
    :param w: transform matrix from hidden layer to output layer
    :return: delta_w, delta_theta, delta_v, delta_gamma
    """
    g = y_hat * (1 - y_hat) * (y - y_hat)
    e = b * (1 - b) * np.dot(np.transpose(w), g)
    delta_w = np.array(eta * np.matrix(g).T * np.matrix(b))
    delta_theta = -eta * g
    delta_v = np.array(eta * np.matrix(e).T * np.matrix(x))
    delta_gamma = -eta * e
    return delta_w, delta_theta, delta_v, delta_gamma

if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description='Back propagation neural networks algorithm')
    parser.add_argument('-e', '--eta', dest='eta', nargs='?', default=0.5, type=float, help='eta (learning rate) value')
    parser.add_argument('--hidden', dest='hidden', nargs='?', default=250, type=int,
                        help='size of hidden layer')
    parser.add_argument('-i', '--iter', dest='iter', nargs='?', default=100, type=int,
                        help='maximum iteration before stop training')
    parser.add_argument('--err', '--error', dest='error', nargs='?', default=10, type=int,
                        help='desired training error rate')
    parser.add_argument('-t', '--train', dest='train', nargs='?', default='', type=str, help='train result filename')
    parser.add_argument('-r', '--result', dest='result', nargs='?', default='', type=str, help='test result filename')
    parser.add_argument('-j', '--jobs', dest='jobs', nargs='?', default=1, type=int, help='maximum parallel jobs allowed')
    args = parser.parse_args()

    timer = UniversalTimer()

    print("Loading training set...")
    labelSpace = list(range(10))
    trainingData, trainingLabel = unpack_mnist('Data/train-images.idx3-ubyte', 'Data/train-labels.idx1-ubyte')

    #trainingData = trainingData[0:2000]
    #trainingLabel = trainingLabel[0:2000]

    t_mean = np.mean(trainingData)
    t_std = np.std(trainingData)

    trainingData -= t_mean
    trainingData /= t_std
    labelSpace = np.array(labelSpace)

    # Standard BP
    v = np.random.rand(args.hidden, trainingData.shape[1])
    w = np.random.rand(labelSpace.shape[0], args.hidden)
    gamma = np.zeros(args.hidden)
    theta = np.zeros(labelSpace.shape[0])
    y = np.zeros((trainingData.shape[0], labelSpace.shape[0]))
    y[np.array(range(trainingLabel.shape[0])), trainingLabel] = 1

    round = 0
    training_error_rate = 100
    while training_error_rate > args.error and round < args.iter:
        # train training set once
        train_error = 0
        for i, x in enumerate(tqdm(trainingData, total=trainingData.shape[0], ascii=True, ncols=75, leave=False)):
            b, y_hat = feed_forward(x, v, w, gamma, theta)
            if np.argmax(y_hat) != trainingLabel[i]:
                train_error += 1
            delta_w, delta_theta, delta_v, delta_gamma = standard_back_propagation(args.eta, x, y[i], b, y_hat, w)
            w += delta_w
            theta += delta_theta
            v += delta_v
            gamma += delta_gamma
        '''
        # if some of training set were assigned to test the training result
        result = list(map(lambda x: feed_forward(x, v, w, gamma, theta), trainingData))
        train_test_error = 0
        for i, r in enumerate(result):
            if np.argmax(r[1]) != trainingLabel[i]:
                train_test_error += 1
        '''
        round += 1
        training_error_rate = train_error / trainingData.shape[0] * 100
        tqdm.write("round %d, training error rate %.2f%%" % (round, training_error_rate))
        if args.train:
            with open(args.train, 'a') as test_result_file:
                # TODO: if training set is split into two parts, update train test error here
                test_result_file.write("%d, %.2f, %.2f\n" % (round, training_error_rate, 0))

    print("Loading testing set...")
    testingData, testingLabel = unpack_mnist('Data/t10k-images.idx3-ubyte', 'Data/t10k-labels.idx1-ubyte')
    testingData -= t_mean
    testingData /= t_std
    result = list(map(lambda x: feed_forward(x, v, w, gamma, theta), testingData))
    test_error = 0
    for i, r in enumerate(result):
        if np.argmax(r[1]) != testingLabel[i]:
            test_error += 1

    print_string = "Classification completed.\n" \
                   "Runtime = %.2f minutes\n" \
                   "Training samples = %d\n" \
                   "Training error rate = %2.2f%%\n" \
                   "Testing samples = %d\n" \
                   "Errors = %d\n" \
                   "Error rate = %2.2f%%\n"\
                   % (timer.current_time() / 60, trainingLabel.shape[0], training_error_rate, testingLabel.shape[0], 
                      test_error, test_error / testingLabel.shape[0] * 100)
    if args.result != '':
        with open(args.result, 'w') as test_result_file:
            test_result_file.write(print_string)
    print('\n', print_string)
