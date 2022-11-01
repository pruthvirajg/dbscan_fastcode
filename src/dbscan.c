#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

#include <x86intrin.h>
#include <immintrin.h>

#include "../include/dbscan.h"
#include "../include/utils.h"
#include "../include/config.h"
#include "../include/queue.h"


static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


float ref_distance(DTYPE_OBS i, DTYPE_OBS j )
{
   float sum = 0.0;
   dst_call_count += 1;

   dst_st = rdtsc();
   for ( int feature = 0 ; feature < FEATURES ; feature++ )
   {
      sum += SQR( ( dataset[ i ].features[ feature ] - dataset[ j ].features[ feature ] ) );
   }
   dst_et = rdtsc();
   
   dst_cycles += (dst_et - dst_st);

   return sqrt( sum );
}


neighbors_t *ref_find_neighbors(DTYPE_OBS observation )
{
   #ifdef DEBUG
   printf("find_neighbours\n");
   #endif
   
   neighbors_t *neighbors = ( neighbors_t * )malloc( sizeof( neighbors_t ) );
   neighbors->neighbor = ( int * )malloc( sizeof(int) * TOTAL_OBSERVATIONS);

   // bzero( (void *)neighbor, sizeof( neighbors_t ) );
   neighbors->neighbor_count = 0;
   bzero((void *)neighbors->neighbor, sizeof(int) * TOTAL_OBSERVATIONS);

   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      if ( i == observation ) continue;

      if ( ref_distance( observation, i ) <= EPSILON )
      {
         neighbors->neighbor[ i ] = 1;
         neighbors->neighbor_count++;
      }
   }

   return neighbors;
}


void ref_free_neighbors( neighbors_t *neighbors )
{
   free( neighbors->neighbor );
   free( neighbors );

   return;
}

void ref_fold_neighbors( neighbors_t *seed_set, neighbors_t *neighbors )
{
   #ifdef DEBUG
   printf("fold_neighbors\n");
   #endif

   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      if ( neighbors->neighbor[ i ] )
      {
         seed_set->neighbor[ i ] = 1;
      }
   }

   return;
}


void ref_process_neighbors( int initial_point, neighbors_t *seed_set )
{
   #ifdef DEBUG
   printf("process_neighbors\n");
   #endif

   // Process every member in the seed set.
   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      // Is this a neighbor?
      if ( seed_set->neighbor[ i ] )
      {
         seed_set->neighbor[ i ] = 0;

         
         if ( dataset[ i ].label != UNDEFINED || dataset[ initial_point ].label==NOISE)
         {
            // already labelled or initial_point is already NOISE, skip
            continue;
         }
         else if ( dataset[ i ].label == NOISE )
         {
            // override NOISE with label
            dataset[ i ].label = dataset[ initial_point ].label;
         }
         else{
            // base case (is UNDEFINED) apply label
            dataset[ i ].label = dataset[ initial_point ].label;
         }

         #ifdef DEBUG
         printf("dataset[%llu]: %s, label:%d\n", i, dataset[i].name, dataset[i].label);
         #endif

         neighbors_t *neighbors = ref_find_neighbors( i );

         if ( neighbors->neighbor_count >= MINPTS )
         {
            ref_fold_neighbors( seed_set, neighbors );
            i = 0;
         }
         ref_free_neighbors( neighbors );
      }
   }

   return;
}


int ref_dbscan( void )
{
   /***
    * Ref impl of DBSCAN
   */
   int cluster = 0;

   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      #ifdef DEBUG
      printf("Working on %llu,  %s\n", i, dataset[i].name);
      #endif
      if ( dataset[ i ].label != UNDEFINED ) continue;
      
      neighbors_t *neighbors = ref_find_neighbors( i );

      if ( neighbors->neighbor_count < MINPTS )
      {
         dataset[ i ].label = NOISE;
         ref_free_neighbors( neighbors );
         continue;
      }

      // Create a new cluster.
      dataset[ i ].label = ++cluster;
      
      ref_process_neighbors( i, neighbors  );

      ref_free_neighbors( neighbors );
   }

   return cluster;
}


int acc_dbscan( void )
{
   int clusters = 0;

   /***
    * Re-written schedule for DBSCAN to support acceleration
   */

   // Calculate the distance and generate epsilon matrix
   // TODO: Replace this with accelerated distance calc kernel
   gen_epsilon_matrix();

   #ifdef DUMP_EPSILON_MAT
   FILE *fp;
   fp = fopen("./epsilon_matrix.csv", "w");
   char buffer[10000];
   char template[] = "%d,";

   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      for (DTYPE_OBS j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         sprintf(buffer, template, epsilon_matrix[i*TOTAL_OBSERVATIONS + j]);
         fputs(buffer, fp);
      }
      fputs("\n", fp);
   }

   fclose(fp);
   #endif

   // Reduction along the rows to check if row has > MIN_PTS
   min_pts_check();
   
   // Label all points
   clusters = class_label();
   return clusters;
}


bool acc_distance(DTYPE_OBS i, DTYPE_OBS j )
{
   // reference implementation for SIMD distance
   float distance = 0.0;
   dst_call_count += 1;
   int res = 0;

   dst_st = rdtsc();
   for ( int feature = 0 ; feature < FEATURES ; feature++ )
   {
      distance += SQR( ( dataset[ i ].features[ feature ] - dataset[ j ].features[ feature ] ) );
   }
   dst_et = rdtsc();
   
   dst_cycles += (dst_et - dst_st);

   res = distance <= EPSILON_SQUARE;
   return res;
}


void gen_epsilon_matrix(void){
   // Calculate the distance and generate square epsilon matrix
   // ASSUMPTION calculating for all pairs of points including itself
   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      for (DTYPE_OBS j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         #ifdef DEBUG
         printf("Working on %llu,  %s\n", i, dataset[i].name);
         #endif

         epsilon_matrix[i*TOTAL_OBSERVATIONS + j] = acc_distance(i, j);
      }
   }
}


void min_pts_check(void){
  DTYPE_OBS num_valid_points = 0;
   
   // Reduction along the rows to check if row has > MIN_PTS
   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      // For each row check if MIN_PTS is met
      num_valid_points = 0;

      for (DTYPE_OBS j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         if(epsilon_matrix[i*TOTAL_OBSERVATIONS + j] == 1){
            num_valid_points+=1;
         }
      }
      
      min_pts_vector[i] = (num_valid_points >= MINPTS) ? true: false;
   }
}


int class_label(void){
   // Label all points

   int cluster = 0;
   int core_pt_label = UNDEFINED;

   // For each entry in min_pts_vector
   for(DTYPE_OBS i=0; i< TOTAL_OBSERVATIONS; i++){
      if(min_pts_vector[i] == false) continue;
      core_pt_label = dataset[i].label != NOISE ? dataset[i].label: ++cluster;
      dataset[i].label = core_pt_label;
      
      // labelling all (direct density reachable) neighbours of core point
      traverse_row(i, cluster, core_pt_label);
      
   }

   return cluster;
}

void traverse_row(DTYPE_OBS row_index, int cluster, int core_pt_label){
   // if any row in epsilon matrix meets criteria, then iterate over the row in epsilon matrix
   // maintain a queue of neighbours that are 1, visit those and label neighbour of neighbours
   // for every row completely visited, set the associated visited(?) entry to 0
   queue_t *N_List = queue_new();

   for(int j=0; j< TOTAL_OBSERVATIONS; j++){
      if (dataset[j].label != NOISE) continue;

      // Assign neighbout to the cluster
      if(epsilon_matrix[row_index*TOTAL_OBSERVATIONS + j]){
         dataset[j].label = core_pt_label;
      }
   }

   queue_free(N_List);
}

void emit_classes(int clusters){
   DTYPE_OBS TOTAL_OBSERVATIONS = OBSERVATIONS * AUGMENT_FACTOR;

   for ( int class = 1 ; class <= clusters ; class++ )
   {
      printf( "Class %d:\n", class );
      for (DTYPE_OBS obs = 0 ; obs < TOTAL_OBSERVATIONS ; obs++ )
      {
         if ( dataset[ obs ].label == class )
         {
            printf("  %s (%d)\n", dataset[ obs ].name, dataset[ obs ].class );
         }
      }
      printf("\n");
   }
}


void emit_outliers(){
   printf( "NOISE\n" );
   DTYPE_OBS TOTAL_OBSERVATIONS = OBSERVATIONS * AUGMENT_FACTOR;
   for (DTYPE_OBS obs = 0 ; obs < TOTAL_OBSERVATIONS ; obs++ )
   {
      if ( dataset[ obs ].label == NOISE )
      {
         printf("  %s (%d)\n", dataset[ obs ].name, dataset[ obs ].class );
      }
   }
   printf("\n");
}

