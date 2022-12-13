CC = gcc
CFLAGS = -I. -std=c99 -Wall -O3 -mavx -mavx2 -mfma -march=native

all: augment main run

augment: augment.c ./src/utils.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

main: main.c ./src/dbscan.c ./src/utils.c ./src/queue.c ./src/acc_distance.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

run:
	./augment
	./main 1

assemble:
	objdump -s -d -f --source ./main > ./odump/main.S
	
clean:
	rm -f main augment ./data/augmented_dataset.csv
	# rm -f ./results/*
	rm -f ./odump/*
