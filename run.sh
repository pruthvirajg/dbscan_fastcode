#!/bin/bash
make clean
make
./augment
./main 1 > ./results/results.txt 