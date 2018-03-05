# README

## 运行环境

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

## 简便方法

运行autorun.bat，分别在MNIST和CIFAR-10数据集上运行Naive Bayes、SVM算法（kNN运行时间过长，没有放入）。

运行autorun-bp.bat，在MNIST数据集上运行BP算法，分别取隐层节点数为50、100、250进行测试，设定迭代轮次为100，测试结果保存在``result``目录下。

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

示例：

对MNIST数据集运行kNN算法，K=5，并行数为4，将结果存储在result/knn-mnist-5.txt

```
python kNN.py -k 5 -d mnist -r result/knn-mnist-5.txt -j 4
```

对CIFAR数据集运行kNN算法，K=1，并行数为4，不存储结果

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

示例：

对MNIST数据集运行Naive Bayes算法，对输入的样本进行二值化处理，并行数4，结果存储在result/mnist-bayes.txt

```
python NaiveBayes.py -d mnist -t -j 4 -r result/mnist-bayes.txt
```

对CIFAR数据集运行Naive Bayes算法，不进行二值化处理，并行数4，不存储结果

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

示例：

对MNIST数据集运行SVM算法，使用200个训练样本和100个测试样本，最大迭代次数=50，rbf核，C=20，$\sigma$=10，训练模型存入result/mnist-svm.json，测试结果存入result/mnist-svm.txt，并行数=4

```
python SVM.py -d mnist -s 200 -t 100 -m 50 -k rbf -c 20 -x 10 -o result/mnist-svm.json -r result/mnist-svm.txt -j 4
```

对CIFAR数据集运行SVM算法，，使用200个训练样本和100个测试样本，最大迭代次数=50，使用线性核，C=1，并行数=4训练模型和结果不保存。

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

示例：

对MNIST数据集运行BP算法，取学习率eta=1，隐层节点数=100，最大迭代轮数=100，目标测试错误率=0（一直迭代，直到达到设定的100轮）。将训练记录以csv格式保存在``result/bp-eta1-hidden100.csv``中，测试结果保存在``result/bp-eta1-hidden100.txt``中。

```
python BPNetwork.py -e 1 --hidden 100 -i 100 --err 0 -t result/bp-eta1-hidden100.csv -r result/bp-eta1-hidden100.txt
```

