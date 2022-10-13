#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

#include "../include/dbscan.h"
#include "../include/utils.h"
#include "../include/config.h"


double distance( int i, int j )
{
   double sum = 0.0;

   for ( int feature = 0 ; feature < FEATURES ; feature++ )
   {
      sum += SQR( ( dataset[ i ].features[ feature ] - dataset[ j ].features[ feature ] ) );
   }

   return sqrt( sum );
}


neighbors_t *find_neighbors( unsigned long observation )
{
   neighbors_t *neighbor = ( neighbors_t * )malloc( sizeof( neighbors_t ) );
   neighbor->neighbor = ( int * )malloc( sizeof(int) * TOTAL_OBSERVATIONS);

   bzero( (void *)neighbor->neighbor, sizeof( neighbors_t ) );

   for ( unsigned long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      if ( i == observation ) continue;

      if ( distance( observation, i ) <= EPSILON )
      {
         neighbor->neighbor[ i ] = 1;
         neighbor->neighbor_count++;
      }
   }

   return neighbor;
}


void free_neighbors( neighbors_t *neighbors )
{
   free( neighbors->neighbor );
   free( neighbors );

   return;
}

void fold_neighbors( neighbors_t *seed_set, neighbors_t *neighbors )
{
   for ( unsigned long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
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
   
   // Process every member in the seed set.
   for ( unsigned long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
      // Is this a neighbor?
      if ( seed_set->neighbor[ i ] )
      {
         seed_set->neighbor[ i ] = 0;

         if ( dataset[ i ].label == NOISE )
         {
            dataset[ i ].label = dataset[ initial_point ].label;
         }
         else if ( dataset[ i ].label != UNDEFINED )
         {
            continue;
         }

         dataset[ i ].label = dataset[ initial_point ].label;

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

   for ( unsigned long i = 0 ; i < TOTAL_OBSERVATIONS ; i++ )
   {
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

