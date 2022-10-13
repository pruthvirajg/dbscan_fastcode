#ifndef _dbscan_h
#define _dbscan_h

#include "../include/config.h"

typedef struct dataset_t {
   char *name;
   int  features[ FEATURES ];
   int  class;
   int  label;
} dataset_t;

typedef struct neighbors_t {
   int neighbor_count;
   // int neighbor[ OBSERVATIONS ];
   int *neighbor;
} neighbors_t;

#define SQR( x )    ( ( x ) * ( x ) )

// Features are:
//    [ 0] hair
//    [ 1] feathers 
//    [ 2] eggs 
//    [ 3] milk
//    [ 4] airborne
//    [ 5] aquatic
//    [ 6] predator
//    [ 7] toothed
//    [ 8] backbone
//    [ 9] breathes
//    [10] venomous
//    [11] fins
//    [12] legs
//    [13] tail
//    [14] domestic
//    [15] catsize

// Classes are [1] Mammal, [2] Bird, [3] Reptile, [4] Fish
//             [5] Amphibian, [6] Bug, [7] Invertebrate.

// Example dataset
//  Name        Features                           Class
// {"aardvark", {1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1}, 1, 0},
// {"antelope", {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
// {"bass",     {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
dataset_t *dataset;

unsigned long TOTAL_OBSERVATIONS;

double distance( int i, int j );

// neighbors_t *find_neighbors( int observation );
neighbors_t *find_neighbors( unsigned long observation );

void free_neighbors( neighbors_t *neighbors );

void fold_neighbors( neighbors_t *seed_set, neighbors_t *neighbors );

void process_neighbors( int initial_point, neighbors_t *seed_set );

int dbscan( void );

void emit_classes(int clusters);

void emit_outliers();

#endif
