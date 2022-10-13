#!/bin/bash
make clean
make
./augment
./main > ./results/results.txt 