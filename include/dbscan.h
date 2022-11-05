#ifndef _dbscan_h
#define _dbscan_h

#include "../include/config.h"
#include <stdbool.h>

// #define VERIFY_ACC 1
// #define DUMP_EPSILON_MAT 1
// #define DEBUG_ACC_DIST 1

bool ACC_DBSCAN;

typedef struct dataset_t {
   char *name;
   DTYPE  features[ FEATURES ];
   int  class;
   int  label;
} dataset_t;

typedef struct neighbors_t {
   unsigned long long neighbor_count;
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

// Globals for distance performance
int dst_call_count;
unsigned long long dst_st;
unsigned long long dst_et;
unsigned long long dst_cycles;

DTYPE_OBS TOTAL_OBSERVATIONS;

float EPSILON_SQUARE;

bool *epsilon_matrix;
bool *ref_epsilon_matrix;

bool *ref_min_pts_vector;
bool *min_pts_vector;

bool *traverse_mask;

float ref_distance(DTYPE_OBS i, DTYPE_OBS j );

neighbors_t *ref_find_neighbors(DTYPE_OBS observation );

void ref_free_neighbors( neighbors_t *neighbors );

void ref_fold_neighbors( neighbors_t *seed_set, neighbors_t *neighbors );

void ref_process_neighbors( int initial_point, neighbors_t *seed_set );

int ref_dbscan( void );

// Functions for accelerated DBSCAN
int acc_dbscan( void );

bool acc_distance(DTYPE_OBS i, DTYPE_OBS j );

void gen_epsilon_matrix(void);

int verify_eps_mat(void);

// Min Points Functions
void calc_min_pts(void);

void acc_min_pts(void);

int verify_min_pts(void);

// Class labelling
void traverse_row(DTYPE_OBS row_index, int cluster, int core_pt_label);

int class_label(void);

// Utilities
void emit_classes(int clusters);

void emit_outliers();

#endif
