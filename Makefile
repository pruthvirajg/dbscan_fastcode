CC = gcc
CFLAGS = -I. -g -std=c99 -Wall -O3

all: augment main assemble

augment: augment.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

main: main.c ./src/dbscan.c ./src/utils.c ./src/queue.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

run:
	./augment
	./main

assemble:
	objdump -s -d -f --source ./main > ./odump/main.S
	
clean:
	rm -f main augment ./data/augmented_dataset.csv
	rm -f ./results/*
	rm -f ./odump/*
