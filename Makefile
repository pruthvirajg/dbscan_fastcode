CC = gcc
CFLAGS = -O3 -I. -g -std=c99 -Wall

all: dbscan assemble

dbscan: dbscan.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

assemble:
	objdump -s -d -f --source ./dbscan > dbscan.S

clean:
	rm -f dbscan
