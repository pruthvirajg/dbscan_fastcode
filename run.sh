#!/bin/bash
make clean
make
./augment
./main > ./results/results_baseline.txt
./main 1 > ./results/results_acc.txt