CC = gcc
CFLAGS = -I. -g -std=c99 -Wall

all: augment main

augment: augment.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

main: main.c ./src/dbscan.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

run:
	./augment
	./main

clean:
	rm -f main augment ./data/augmented_dataset.csv
