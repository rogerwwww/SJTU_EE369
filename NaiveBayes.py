#!python3
from utility import *
import numpy as np


class GaussianClassifier(object):
    """
    Gaussian classifier decorating class. It is a trick to solve the problem that lambda can't be pickled.
    See https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function
    """
    def __init__(self, mus, sigmas, Pc):
        self.mus = mus
        self.sigmas = sigmas
        self.Pc = Pc

    def __call__(self, x):
        try:
            ret = classify_naive_bayes(x, self.mus, self.sigmas, self.Pc)
        except KeyboardInterrupt:
            return
        return ret


class BinaryClassifier(object):
    """
    Binary classifier decorating class. It is a trick to solve the problem that lambda can't be pickled.
    """
    def __init__(self, priors, Pc):
        self.priors = priors
        self.Pc = Pc

    def __call__(self, x):
        try:
            ret = classify_binary_naive_bayes(x, self.priors, self.Pc)
        except KeyboardInterrupt:
            return
        return ret


def train_naive_bayes(training_set, label_set, label_space, type='continuous'):
    """
    Naive bayes trainer
    :param training_set: 
    :param label_set: 
    :param label_space: 
    :param type: bayes classifier type. continuous(default) or binary
    :return: mus, sigmas, Pc (for continuous); priors, Pc(for binary)
    """
    training_set = np.array(training_set)
    data_numbers = training_set.shape[0]
    data_length = training_set.shape[1]
    label_set = np.array(label_set)
    label_numbers =len(label_space)
    training_set, label_set = sort_data_by_label(training_set, label_set)

    Pc = np.empty((label_numbers, 1))
    training_c = []
    prev_split = 0
    for i in range(label_numbers):
        split = np.searchsorted(label_set, i, side='right') + 1
        training_c.append(training_set[prev_split: split])
        Pc[i] = (split - prev_split + 1) / (data_numbers + label_numbers)
        prev_split = split

    if type == 'continuous':
        mus = map(mle_mu, training_c)
        mus = np.array(list(mus))
        sigmas = map(mle_sigma, training_c)
        sigmas = np.array(list(sigmas))
        return mus, sigmas, Pc
    elif type == 'binary':
        priors = map(calc_binary_Pxic, training_c)
        priors = list(priors)
        return priors, Pc


def classify_naive_bayes(testing_data, mus, sigmas, Pc):
    """
    Classify a single testing data to an index in continuous bayes classification
    :param testing_data: 
    :param mus: 
    :param sigmas: 
    :param Pc: 
    :return: classified index
    """
    mus_list = list(mus)
    sigmas_list = list(sigmas)
    Pxic = map(lambda mu, sigma: log_gaussian(testing_data, mu, sigma), mus_list, sigmas_list)
    Pxic = np.array(list(Pxic))
    Pxc = np.nansum(Pxic, axis=1)
    P = np.mat(Pxc).T + np.log(Pc)
    index = np.argmax(P)
    return index


def classify_binary_naive_bayes(testing_data, priors, Pc):
    """
    Classify a single testing data to an index in binary bayes classification
    :param testing_data: 
    :param priors: 
    :param Pc: 
    :return: classified index
    """
    Pxic = map(lambda x: np.sum(x * [1 - testing_data, testing_data], axis=0), priors)
    Pxic = np.array(list(Pxic))
    Pxc = np.sum(np.log(Pxic), axis=1)
    P = np.mat(Pxc).T + np.log(Pc)
    index = np.argmax(P)
    return index


def mle_mu(training_set):
    """
    Estimate Gaussian mu(expectation) using maximum likelihood estimation
    :param training_set: 
    :return: estimated mu
    """
    return list(np.mean(training_set, axis=0))


def mle_sigma(training_set):
    """
    Estimate Gaussian sigma(standard deviation) using maximum likelihood estimation
    :param training_set: 
    :return: estimated sigma
    """
    return list(np.std(training_set, axis=0))


def log_gaussian(x, mu, sigma):
    """
    Calculate log(gaussian(x)) with all parameters given
    :param x: input x
    :param mu: expectation
    :param sigma: standard deviation
    :return: calculated result
    """
    return - (x - mu) ** 2 / (2 * sigma ** 2) - np.log(np.sqrt(2 * np.pi) * sigma)


def calc_binary_Pxic(training_set):
    """
    Calculate Pxic in binary bayes training
    :param training_set: 
    :return: Pxic
    """
    p1 = np.count_nonzero(training_set, axis=0)
    p0 = training_set.shape[0] - p1
    p0 = (p0 + 1) / (training_set.shape[0] + 2)
    p1 = (p1 + 1) / (training_set.shape[0] + 2)
    return np.array([p0, p1])


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description='Naive Bayse on MNIST and CIFAR data set.')
    parser.add_argument('-t', '--threshold', dest='threshold', action='store_const', const=True,
                        help='whether to threshold image')
    parser.add_argument('-d', '--data', dest='dataset', default='cifar', choices=['cifar', 'mnist'], nargs='?', type=str,
                        help='Indicate which data set to use (mnist or cifar), default=cifar')
    parser.add_argument('-r', '--result', dest='result', nargs='?', default='', type=str, help='test result filename')
    parser.add_argument('-j', '--jobs', dest='jobs', nargs='?', default=1, type=int, help='maximum parallel jobs allowed')

    args = parser.parse_args()

    timer = UniversalTimer()

    print("Loading training data...")
    if args.dataset == 'mnist':
        # Load training set
        labelSpace = list(range(10))
        trainingData, trainingLabel = unpack_mnist('Data/train-images.idx3-ubyte', 'Data/train-labels.idx1-ubyte')
    else:
        labelSpace = CIFAR_LABEL_SPACE
        trainingData, trainingLabel = \
            unpack_cifar_feature('CIFAR-10/HOG+SVM classifier/data/features/train')

    print("Training...")
    if args.threshold:
        # Threshold training set and train discrete(binary) Bayes
        if args.dataset == 'mnist':
            trainingData = threshold(trainingData)
        else:
            trainingData = threshold(trainingData, thres=0.02)
        priors, Pc = train_naive_bayes(trainingData, trainingLabel, labelSpace, type='binary')
    else:
        # Train continuous(Gaussian) Bayes
        mus, sigmas, Pc = train_naive_bayes(trainingData, trainingLabel, labelSpace)

    print("Loading test data...")
    if args.dataset == 'mnist':
        # Load test data
        testingData, testingLabel = unpack_mnist('Data/t10k-images.idx3-ubyte', 'Data/t10k-labels.idx1-ubyte')
        testingData = list(testingData)
    else:
        testingData, testingLabel = \
            unpack_cifar_feature('CIFAR-10/HOG+SVM classifier/data/features/test')

    print("Classifying test data...")
    with Pool(args.jobs) as p:
        if args.threshold:
            # Test binary result
            if args.dataset == 'mnist':
                testingData = threshold(testingData)
            else:
                testingData = threshold(testingData, thres=0.02)
            classifiedLabel_it = p.imap(BinaryClassifier(priors, Pc), testingData)
        else:
            # Test gaussian result
            classifiedLabel_it = p.imap(GaussianClassifier(mus, sigmas, Pc), testingData)

        classifiedLabel = np.array(list(tqdm(classifiedLabel_it, total=testingData.shape[0], ascii=True, ncols=75)))

    error = np.count_nonzero(classifiedLabel - testingLabel)

    print_string = "Classification completed.\n" \
                   "Runtime = %.2f minutes\n" \
                   "Training samples = %d\n" \
                   "Testing samples = %d\n" \
                   "Errors = %d\n" \
                   "Error rate = %2.4f%%\n"\
                   % (timer.current_time() / 60, trainingLabel.shape[0], testingLabel.shape[0], error,
                      error / testingLabel.shape[0] * 100)
    if args.result != '':
        with open(args.result, 'w') as test_result_file:
            test_result_file.write(print_string)
    print('\n', print_string)
