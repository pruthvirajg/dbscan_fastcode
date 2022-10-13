CC = gcc
CFLAGS = -I. -g -std=c99 -Wall

all: main

main: main.c ./src/dbscan.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

clean:
	rm -f main
