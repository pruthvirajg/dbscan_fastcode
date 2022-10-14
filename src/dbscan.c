#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

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


double distance( unsigned long long i, unsigned long long j )
{
   double sum = 0.0;
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


neighbors_t *find_neighbors( unsigned long long observation )
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

      if ( distance( observation, i ) <= EPSILON )
      {
         neighbors->neighbor[ i ] = 1;
         neighbors->neighbor_count++;
      }
   }

   return neighbors;
}


void free_neighbors( neighbors_t *neighbors )
{
   free( neighbors->neighbor );
   free( neighbors );

   return;
}

void fold_neighbors( neighbors_t *seed_set, neighbors_t *neighbors )
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


void process_neighbors( int initial_point, neighbors_t *seed_set )
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

         neighbors_t *neighbors = find_neighbors( i );

         if ( neighbors->neighbor_count >= MINPTS )
         {
            fold_neighbors( seed_set, neighbors );
            i = 0;
         }
         free_neighbors( neighbors );
      }
   }

   return;
}


int dbscan( void )
{
   int cluster = 0;

   for ( unsigned long long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      #ifdef DEBUG
      printf("Working on %llu,  %s\n", i, dataset[i].name);
      #endif
      if ( dataset[ i ].label != UNDEFINED ) continue;
      
      neighbors_t *neighbors = find_neighbors( i );

      if ( neighbors->neighbor_count < MINPTS )
      {
         dataset[ i ].label = NOISE;
         free_neighbors( neighbors );
         continue;
      }

      // Create a new cluster.
      dataset[ i ].label = ++cluster;
      
      process_neighbors( i, neighbors  );

      free_neighbors( neighbors );
   }

   return cluster;
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

