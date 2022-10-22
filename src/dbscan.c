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


static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


float ref_distance( unsigned long long i, unsigned long long j )
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


neighbors_t *ref_find_neighbors( unsigned long long observation )
{
   #ifdef DEBUG
   printf("find_neighbours\n");
   #endif
   
   neighbors_t *neighbors = ( neighbors_t * )malloc( sizeof( neighbors_t ) );
   neighbors->neighbor = ( int * )malloc( sizeof(int) * TOTAL_OBSERVATIONS);

   // bzero( (void *)neighbor, sizeof( neighbors_t ) );
   neighbors->neighbor_count = 0;
   bzero((void *)neighbors->neighbor, sizeof(int) * TOTAL_OBSERVATIONS);

   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
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

   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
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
   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
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

   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
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

   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      for (unsigned long long j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
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
   // class_label();
   return clusters;
}


int acc_distance( unsigned long long i, unsigned long long j )
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
   // Calculate the distance and generate epsilon matrix
   // ASSUMPTION calculating for all pairs of points including itself
   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      for ( unsigned long long j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         #ifdef DEBUG
         printf("Working on %llu,  %s\n", i, dataset[i].name);
         #endif

         epsilon_matrix[i*TOTAL_OBSERVATIONS + j] = acc_distance(i, j);
      }
   }
}


void min_pts_check(void){
   unsigned long long num_valid_points = 0;

   // Reduction along the rows to check if row has > MIN_PTS
   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      // For each row check if MIN_PTS is met
      num_valid_points = 0;

      for (unsigned long long j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         if(epsilon_matrix[i*TOTAL_OBSERVATIONS + j] == 1){
            num_valid_points++;
         }
      }

      min_pts_vector[i] = (num_valid_points >= MINPTS) ? true: false;
   }
}


void class_label(void){
   // Label all points
}


void emit_classes(int clusters){
   unsigned long TOTAL_OBSERVATIONS = OBSERVATIONS * AUGMENT_FACTOR;

   for ( int class = 1 ; class <= clusters ; class++ )
   {
      printf( "Class %d:\n", class );
      for ( unsigned long obs = 0 ; obs < TOTAL_OBSERVATIONS ; obs++ )
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
   unsigned long TOTAL_OBSERVATIONS = OBSERVATIONS * AUGMENT_FACTOR;
   for ( unsigned long obs = 0 ; obs < TOTAL_OBSERVATIONS ; obs++ )
   {
      if ( dataset[ obs ].label == NOISE )
      {
         printf("  %s (%d)\n", dataset[ obs ].name, dataset[ obs ].class );
      }
   }
   printf("\n");
}

