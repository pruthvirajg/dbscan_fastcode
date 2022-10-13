CC = gcc
CFLAGS = -I. -g -std=c99 -Wall

all: augment main assemble

augment: augment.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

main: main.c ./src/dbscan.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

run:
	./augment
	./main

assemble:
	objdump -s -d -f --source ./dbscan > dbscan.S
	
clean:
	rm -f main augment ./data/augmented_dataset.csv
