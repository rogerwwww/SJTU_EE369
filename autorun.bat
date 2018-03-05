@echo off
echo ML Project Autorun
echo.
echo Naive Bayes on MNIST
python NaiveBayes.py -d mnist -t -j 4 -r result/mnist-bayes.txt
echo Result stored in result/mnist-bayes.txt
echo.
echo SVM on MNIST
python SVM.py -d mnist -s 200 -t 100 -m 50 -k rbf -c 20 -x 10 -o result/mnist-svm.json -r result/mnist-svm.txt -j 4
echo Model stored in result/mnist-svm.json
echo Result stored in result/mnist-svm.txt
echo.
echo Naive Bayes on CIFAR-10
python NaiveBayes.py -d cifar -t -j 4 -r result/cifar-bayes.txt
echo Result stored in result/cifar-bayes.txt
echo.
echo SVM on MNIST
python SVM.py -d cifar -s 200 -t 100 -m 50 -k rbf -c 1 -x 1 -o result/cifar-svm.json -r result/cifar-svm.txt -j 4
echo Model stored in result/cifar-svm.json
echo Result stored in result/cifar-svm.txt
pause