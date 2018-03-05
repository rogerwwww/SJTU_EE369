#!python3
from utility import *
from numpy import matlib
import os
from tqdm import tqdm


class SingleClassSVM:
    def __init__(self, data_set, label_set, C, error_tolerance, max_iteration_times, kernel_type=('lin', 0)):
        self.X = np.mat(data_set)
        self.label_set = np.mat(label_set).T
        self.C = C
        self.error_tolerance = error_tolerance
        self.max_iteration_times = max_iteration_times
        self.kernel_type = kernel_type

        self.data_size = np.shape(data_set)[0]
        self.alphas = np.mat(np.zeros((self.data_size, 1)))
        self.b = 0
        self.w = matlib.zeros(np.shape(data_set)[1])
        # E values are cached for SMO inspiration
        # initial alpha = 0, so E = -y
        self.E_cache = -self.label_set


def clip_alpha(alpha, H, L):
    """
    Restrict alpha value within the boundary of L and H
    :param alpha: input alpha value
    :param H: upper boundary
    :param L: lower boundary
    :return: clipped alpha
    """
    if alpha > H:
        alpha = H
    if L > alpha:
        alpha = L
    return alpha


def kernel(svm_obj, X1, X2):
    """
    Kernel transform function. Note that svm_obj.kernel_type is used to specify the kernel type.
    :param svm_obj: shared svm object
    :param X1: X1 array
    :param X2: X2 array
    :return: kernel transformation result
    """
    m, n = np.shape(X1)
    k = np.mat(np.zeros((m, 1)))
    if svm_obj.kernel_type[0] == 'lin':
        k = X1 * X2.T  # linear kernel
    elif svm_obj.kernel_type[0] == 'rbf':
        for j in range(m):
            delta_row = X1[j, :] - X2
            k[j] = delta_row * delta_row.T
        k = np.exp(k / (-1 * svm_obj.kernel_type[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        print('\nErr: Kernel unknown!')
    return k


def generate_j_random(svm_obj, i):
    """
    Generate a random j value.
    :param svm_obj: shared svm object
    :param i: input i value
    :return: random j value
    """
    j = i  # we want to select any J not equal to i
    while j == i:
        j = int(np.random.uniform(0, svm_obj.data_size))
    return j


def generate_j(svm_obj, i, Ei):
    """
    SMO inspiration function. Given i and Ei, return the best j.
    :param svm_obj: shared svm object
    :param i: first sample index
    :param Ei: E value of the first sample
    :return: j and Ej
    """
    svm_obj.E_cache[i] = Ei

    #  choose the alpha that gives the maximum delta E
    delta_Es = np.abs(svm_obj.E_cache - Ei)
    j = np.argmax(delta_Es)
    Ej = svm_obj.E_cache[j]

    return j, Ej


def calc_Ek(svm_obj, k):
    """
    Calculate E for index k
    :param svm_obj: shared svm object
    :param k: index
    :return: calculated Ek
    """
    fXk = np.multiply(svm_obj.alphas, svm_obj.label_set).T * kernel(svm_obj, svm_obj.X, svm_obj.X[k, :]) + svm_obj.b
    Ek = fXk - svm_obj.label_set[k]
    return Ek


def update_Ek(svm_obj, k):
    """
    Update cached E
    :param k: 
    :param svm_obj: shared svm object
    :return: 
    """
    Ek = calc_Ek(svm_obj, k)
    svm_obj.E_cache[k] = Ek


def smo_inner_loop(svm_obj, i):
    """
    The inner loop of SMO algorithm
    :param svm_obj: shared svm object
    :param i: The index of first sample in sample pairs(i, j)
    :return: whether changed alpha values. 0 = unchanged, 1 = changed (go to function smo to see why)
    """
    Ei = calc_Ek(svm_obj, i)
    if ((svm_obj.label_set[i] * Ei < -svm_obj.error_tolerance) and (svm_obj.alphas[i] < svm_obj.C)) or \
        ((svm_obj.label_set[i] * Ei > svm_obj.error_tolerance) and (svm_obj.alphas[i] > 0)):
        j, Ej = generate_j(svm_obj, i, Ei)
        alpha_i_old = svm_obj.alphas[i].copy()
        alpha_j_old = svm_obj.alphas[j].copy()
        if svm_obj.label_set[i] != svm_obj.label_set[j]:
            L = max(0, svm_obj.alphas[j] - svm_obj.alphas[i])
            H = min(svm_obj.C, svm_obj.C + svm_obj.alphas[j] - svm_obj.alphas[i])
        else:
            L = max(0, svm_obj.alphas[j] + svm_obj.alphas[i] - svm_obj.C)
            H = min(svm_obj.C, svm_obj.alphas[j] + svm_obj.alphas[i])
        if L == H:
            # L==H
            return 0
        eta = 2.0 * kernel(svm_obj, svm_obj.X[i], svm_obj.X[j]) - kernel(svm_obj, svm_obj.X[i], svm_obj.X[i]) - \
            kernel(svm_obj, svm_obj.X[j], svm_obj.X[j])
        if eta >= 0:
            # eta>=0
            return 0
        svm_obj.alphas[j] -= svm_obj.label_set[j] * (Ei - Ej) / eta
        svm_obj.alphas[j] = clip_alpha(svm_obj.alphas[j], H, L)
        update_Ek(svm_obj, j)  # added this for the Ecache
        if abs(svm_obj.alphas[j] - alpha_j_old) < 0.00001:
            # j not moving enough
            return 0
        svm_obj.alphas[i] += svm_obj.label_set[j] * svm_obj.label_set[i] * (alpha_j_old - svm_obj.alphas[j])
        update_Ek(svm_obj, i)
        b1 = svm_obj.b - Ei - svm_obj.label_set[i] * (svm_obj.alphas[i] - alpha_i_old) * kernel(svm_obj, svm_obj.X[i], svm_obj.X[i]) - \
             svm_obj.label_set[j] * (svm_obj.alphas[j] - alpha_j_old) * kernel(svm_obj, svm_obj.X[i], svm_obj.X[j])
        b2 = svm_obj.b - Ej - svm_obj.label_set[i] * (svm_obj.alphas[i] - alpha_i_old) * kernel(svm_obj, svm_obj.X[i], svm_obj.X[j]) - \
             svm_obj.label_set[j] * (svm_obj.alphas[j] - alpha_j_old) * kernel(svm_obj, svm_obj.X[j], svm_obj.X[j])
        if (0 < svm_obj.alphas[i]) and (svm_obj.C > svm_obj.alphas[i]):
            svm_obj.E_cache -= svm_obj.b - b1
            svm_obj.b = b1
        elif (0 < svm_obj.alphas[j]) and (svm_obj.C > svm_obj.alphas[j]):
            svm_obj.E_cache -= svm_obj.b - b2
            svm_obj.b = b2
        else:
            svm_obj.E_cache -= svm_obj.b - (b1 + b2) / 2
            svm_obj.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smo(svm_obj):
    """
    Basic Sequential Minimal Optimization (can only classify two classes)
    :param svm_obj: shared svm object
    :return: b and alphas
    """
    tqdm.write("Starting smo, PID = %d" % os.getpid())
    iterator = 0
    entire_set = True
    alpha_pairs_changed = 0

    while (iterator < svm_obj.max_iteration_times) and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:  # go over all
            for i in range(svm_obj.data_size):
                alpha_pairs_changed += smo_inner_loop(svm_obj, i)
            tqdm.write("fullSet, iterator = %d, pairs changed = %d, pid = %d" % (iterator, alpha_pairs_changed, os.getpid()))
            iterator += 1
        else:  # go over non-bound (railed) alphas
            non_bound_i = np.nonzero((svm_obj.alphas.A > 0) * (svm_obj.alphas.A < svm_obj.C))[0]
            for i in non_bound_i:
                alpha_pairs_changed += smo_inner_loop(svm_obj, i)
            tqdm.write("non-bound, iterator = %d, non_bound = %d, pairs changed = %d, pid = %d" %
                (iterator, len(non_bound_i), alpha_pairs_changed, os.getpid()))
            iterator += 1
        if entire_set:
            entire_set = False  # toggle entire set loop
        elif alpha_pairs_changed == 0:
            entire_set = True
    return svm_obj.b, svm_obj.alphas


def calc_w(svm_obj):
    """
    Calculate w values. Only works with linear kernel
    :param svm_obj: shared svm object
    :return: w matrix
    """
    if svm_obj.kernel_type[0] != 'lin':
        print('Only linear kernel can meaningful w be calculated')
        return None
    X = np.mat(svm_obj.X)
    m, n = np.shape(X)
    w = matlib.zeros((1, n))
    for i in range(m):
        w += svm_obj.alphas[i] * svm_obj.label_set[i] * X[i, :]
    return w


def calc_fx(svm_obj, x):
    """
    Calculate fx value according to input x.
    :param svm_obj: shared svm object
    :param x: input data vector
    :return: fx value
    """
    test_data = matlib.mat(x)
    if svm_obj.kernel_type[0] == 'lin':
        svm_obj.w = calc_w(svm_obj)
        return svm_obj.w * test_data.T + svm_obj.b
    else:
        fx = 0
        for i in range(svm_obj.data_size):
            fx += svm_obj.alphas[i] * svm_obj.label_set[i] * kernel(svm_obj, test_data, svm_obj.X[i, :])
        fx += svm_obj.b
        return fx


def load_training_result(json_file):
    all_dict = read_json(json_file)
    alphas = []
    bs = []
    for svm_dict in sorted(all_dict):
        bs.append(all_dict[svm_dict]['b'])
        alpha_dict = all_dict[svm_dict]['alphas']
        alpha_array = np.zeros(len(alpha_dict))
        for alpha_index in alpha_dict:
            alpha_array[int(alpha_index)] = alpha_dict[alpha_index]
        alphas.append(alpha_array)
    return alphas, bs


if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    import sys

    parser = argparse.ArgumentParser(description='SVM algorithm on MNIST and CIFAR dataset.')
    parser.add_argument('-d --data', dest='dataset', default='cifar', choices=['cifar', 'mnist'], nargs='?', type=str,
                        help='Indicate which data set to use (mnist or cifar)')
    parser.add_argument('-c', dest='C', nargs='?', default=2, type=float, help='C value, default=2')
    parser.add_argument('-e --error', dest='error', nargs='?', default=0.001, type=float,
                        help='error tolerance, default=0.01')
    parser.add_argument('-m --max', dest='max', nargs='?', default=20, type=int,
                        help='max iteration time, default=20')
    parser.add_argument('-k --kernel', dest='kernel', nargs='?', default='rbf', type=str,
                        help='kernel type, default=rbf')
    parser.add_argument('-x --sigma', dest='sigma', nargs='?', default=10, type=float,
                        help='kernel sigma value, default=10')
    parser.add_argument('-s --sample', dest='sample', nargs='?', default=100, type=int,
                        help='sample set numbers, default=100')
    parser.add_argument('-t --test', dest='test', nargs='?', default=100, type=int,
                        help='test set numbers, default=100')
    parser.add_argument('-o --output', dest='output', nargs='?', default='', type=str, help='smo result filename')
    parser.add_argument('-r --result', dest='result', nargs='?', default='', type=str, help='test result filename')
    parser.add_argument('-j --jobs', dest='jobs', nargs='?', default=1, type=int, help='maximum parallel jobs allowed')
    parser.add_argument('-n --normalize', dest='normal',  action='store_const', const=True,
                        help='whether to normalize cifar data')

    args = parser.parse_args()

    timer = UniversalTimer()

    print("Unpacking training data...")
    if args.dataset == 'mnist':
        labelSpace = list(range(10))
        trainingData, trainingLabel = unpack_mnist('Data/train-images.idx3-ubyte', 'Data/train-labels.idx1-ubyte')

        trainingData = trainingData[:args.sample]
        trainingLabel = trainingLabel[:args.sample]
        # normalize training set
        trainingData = trainingData / 255
    else:
        labelSpace = CIFAR_LABEL_SPACE
        trainingData, trainingLabel = \
            unpack_cifar_feature('CIFAR-10/HOG+SVM classifier/data/features/train', max_num=args.sample)
        if args.normal:
            # normalize data
            scaler = 1 / np.max(trainingData, axis=0)
            trainingData *= scaler


    all_svm = []
    for i in range(len(labelSpace)):
        temp_label_set_for_svm = np.zeros(trainingLabel.shape)
        for j in range(trainingLabel.shape[0]):
            if trainingLabel[j] == i:
                temp_label_set_for_svm[j] = 1
            else:
                temp_label_set_for_svm[j] = -1
        all_svm.append(SingleClassSVM(trainingData, temp_label_set_for_svm, args.C, args.error, args.max,
                                      (args.kernel, args.sigma)))

    pool = Pool(args.jobs)
    inPoolSmo = InPoolDeco(smo)
    map_iter = pool.imap(inPoolSmo, all_svm)
    map_result = list(tqdm(map_iter, ascii=True, total=len(all_svm), maxinterval=0.2, ncols=75))

    # update svms
    for i in range(len(map_result)):
        all_svm[i].b = map_result[i][0]
        all_svm[i].alphas = map_result[i][1]

    # store result in json
    if args.output != '':
        data = {}
        i = 0
        for result in map_result:
            index = str(i)
            alpha_array = {}
            for j in range(result[1].shape[0]):
                alpha_array[str(j)] = float(result[1][j])
            single_data = {"alphas": alpha_array, "b": float(result[0])}
            data[index] = single_data
            i += 1
        write_json(args.output, data)

    print("\nTraining completed.")

    print("Loading test data set...")
    if args.dataset == 'mnist':
        testingData, testingLabel = unpack_mnist('Data/t10k-images.idx3-ubyte', 'Data/t10k-labels.idx1-ubyte')
        testingData = testingData[:args.test]
        testingLabel = testingLabel[:args.test]
        # normalize testing set
        testingData = testingData / 255
    else:
        testingData, testingLabel = unpack_cifar_feature(
            'CIFAR-10/HOG+SVM classifier/data/features/test', max_num=args.test)
        if args.normal:
            testingData *= scaler

    print("Testing...")
    error = 0
    for i in tqdm(range(testingData.shape[0]), ascii=True, ncols=75):
        fxs = map(lambda x: calc_fx(x, testingData[i, :]), all_svm)
        fxs = np.array(list(fxs))
        classify_result = np.argmax(fxs)
        str2write = "Classifying testing set %d / %d" % (i, testingData.shape[0]) + "\tClassify result = " + \
                    str(labelSpace[int(classify_result)]) + "\tLabel = " + str(labelSpace[int(testingLabel[i])])
        tqdm.write(str2write)
        if classify_result != testingLabel[i]:
            error += 1

    sys.stdout.write("\n")
    sys.stdout.flush()

    print_string = "Classification completed.\n" \
                   "Runtime = %.2f minutes\n" \
                   "Training samples = %d\n" \
                   "Testing samples = %d\n" \
                   "Errors = %d\n" \
                   "Error rate = %2.4f%%\n" \
                   % (timer.current_time() / 60, trainingLabel.shape[0], testingLabel.shape[0], error,
                      (float(error) / testingLabel.shape[0] * 100))
    if args.result != '':
        with open(args.result, 'w') as test_result_file:
            test_result_file.write(print_string)
    print("\n" + print_string)
