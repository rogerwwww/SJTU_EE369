@echo off
echo BP on MNIST Autorun
python BPNetwork.py -e 1 --hidden 50 -i 100 --err 0 -t result/bp-eta1-hidden50.csv -r result/bp-eta1-hidden50.txt
python BPNetwork.py -e 1 --hidden 100 -i 100 --err 0 -t result/bp-eta1-hidden100.csv -r result/bp-eta1-hidden100.txt
python BPNetwork.py -e 1 --hidden 250 -i 100 --err 0 -t result/bp-eta1-hidden250.csv -r result/bp-eta1-hidden250.txt
pause