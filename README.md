# SJTU_EE369

This repository contains project code for Shanghai Jiao Tong University EE369 (Machine Learning). kNN, Naive Bayes, Support Vector Machine and BP Network on MNIST and CIFAR-10.

## Environment

Windows10 64bit

Python3.6.3

| Package      | Version |
| ------------ | ------- |
| colorama     | 0.3.9   |
| matplotlib   | 2.1.0   |
| numpy        | 1.13.3  |
| scikit-image | 0.13.1  |
| scikit-learn | 0.19.1  |
| scipy        | 1.0.0   |
| tqdm         | 4.19.4  |

## An Easy Approach

Run `autorun.bat`, it will run Naive Bayes and SVM on MNIST and CIFAR-10 dataset. kNN will take a very long time, so it was not included.

Run `autorun-bp.bat`, it will run BP on MNIST dataset. Taking the number of nodes of hidden layer as 50, 100 and 250 and setting the iteration as 100, the result will be stored in ``result`` directory.

Detailed documentation and examples on each algorithm are given below.

## kNN

```
usage: kNN.py [-h] [-k [K]] [-d --data [{cifar,mnist}]] [-r --result [RESULT]]
              [-j --jobs [JOBS]]

kNN method on MNIST and CIFAR data set.

optional arguments:
  -h, --help            show this help message and exit
  -k [K]                k value of kNN, default=3
  -d --data [{cifar,mnist}]
                        Indicate which data set to use (mnist or cifar),
                        default=cifar
  -r --result [RESULT]  test result filename
  -j --jobs [JOBS]      maximum parallel jobs allowed
```

Example：

 Run kNN on MNIST dataset, set `K=5`, `parallel jobs = 4`, the result will be stored in `result/knn-mnist-5.txt`.

```
python kNN.py -k 5 -d mnist -r result/knn-mnist-5.txt -j 4
```

Run kNN on CIFAR, set `K=1`, `parallel jobs = 4`, and don't store the result.

```
python kNN.py -k 1 -d cifar -j 4
```

## Naive Bayes

```
usage: NaiveBayes.py [-h] [-t --threshold] [-d --data [{cifar,mnist}]]
                     [-r --result [RESULT]] [-j --jobs [JOBS]]

Naive Bayse on MNIST and CIFAR data set.

optional arguments:
  -h, --help            show this help message and exit
  -t --threshold        whether to threshold image
  -d --data [{cifar,mnist}]
                        Indicate which data set to use (mnist or cifar),
                        default=cifar
  -r --result [RESULT]  test result filename
  -j --jobs [JOBS]      maximum parallel jobs allowed
```

Example：

Run Naive Bayes on MNIST, binarize (threshold) the input image, set `parallel jobs = 4`. The result will be stored in `result/mnist-bayes.txt`

```
python NaiveBayes.py -d mnist -t -j 4 -r result/mnist-bayes.txt
```

Run Naive Bayes on CIFAR, no binarization, with `parallel jobs = 4`, don't store the result.

```
python NaiveBayes.py -d cifar -j 4
```

## SVM

```
usage: SVM.py [-h] [-d --data [{cifar,mnist}]] [-c [C]] [-e --error [ERROR]]
              [-m --max [MAX]] [-k --kernel [KERNEL]] [-x --sigma [SIGMA]]
              [-s --sample [SAMPLE]] [-t --test [TEST]] [-o --output [OUTPUT]]
              [-r --result [RESULT]] [-j --jobs [JOBS]] [-n --normalize]

SVM algorithm on MNIST and CIFAR dataset.

optional arguments:
  -h, --help            show this help message and exit
  -d --data [{cifar,mnist}]
                        Indicate which data set to use (mnist or cifar)
  -c [C]                C value, default=2
  -e --error [ERROR]    error tolerance, default=0.01
  -m --max [MAX]        max iteration time, default=20
  -k --kernel [KERNEL]  kernel type, default=rbf
  -x --sigma [SIGMA]    kernel sigma value, default=10
  -s --sample [SAMPLE]  sample set numbers, default=100
  -t --test [TEST]      test set numbers, default=100
  -o --output [OUTPUT]  smo result filename
  -r --result [RESULT]  test result filename
  -j --jobs [JOBS]      maximum parallel jobs allowed
  -n --normalize        whether to normalize cifar data
```

Example：

Run SVM on MNIST, with 200 training images and 100 testing images. `max iteration = 50`,  `C = 20` and use `rbf` kernel with $\sigma$=10. The trained model will be stored in `result/mnist-svm.json`, and the test result will be stored in `result/mnist-svm.txt`. `parallel jobs = 4`.

```
python SVM.py -d mnist -s 200 -t 100 -m 50 -k rbf -c 20 -x 10 -o result/mnist-svm.json -r result/mnist-svm.txt -j 4
```

Run SVM on CIFAR, with 200 training images and 100 testing images. `max iteration = 50`,  `C = 1` and use `linear` kernel. No model or result will be stored. `parallel jobs = 4`.

```
python SVM.py -d cifar -s 200 -t 100 -m 50 -k lin -c 1 -j 4
```

## BP Network

```
usage: BPNetwork.py [-h] [-e, --eta [ETA]] [--hidden [HIDDEN]]
                    [-i, --iter [ITER]] [--err, --error [ERROR]]
                    [-t, --train [TRAIN]] [-r, --result [RESULT]]

Back propagation neural networks algorithm

optional arguments:
  -h, --help            show this help message and exit
  -e, --eta [ETA]       eta (learning rate) value
  --hidden [HIDDEN]     size of hidden layer
  -i, --iter [ITER]     maximum iteration before stop training
  --err, --error [ERROR]
                        desired training error rate
  -t, --train [TRAIN]   train result filename
  -r, --result [RESULT]
                        test result filename
```

Example：

Run BP on MNIST. Set the learning rate `eta = 1`, `hidden layer nodes = 100`, `max iteration = 100`, `target error rate = 0` (which means it will run the training routine until it reaches the max iteration 100). The training progress will be stored in ``result/bp-eta1-hidden100.csv``, and test result will be stored in ``result/bp-eta1-hidden100.txt``.

```
python BPNetwork.py -e 1 --hidden 100 -i 100 --err 0 -t result/bp-eta1-hidden100.csv -r result/bp-eta1-hidden100.txt
```

