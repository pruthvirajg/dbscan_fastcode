#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

#define OBSERVATIONS	103
#define FEATURES	16

#define UNDEFINED        0
#define CLASS_1          1
#define CLASS_2          2
#define CLASS_3          3
#define CLASS_4          4
#define CLASS_5          5
#define CLASS_6          6
#define CLASS_7          7
#define NOISE            8

typedef struct dataset_t {
   char *name;
   int  features[ FEATURES ];
   int  class;
   int  label;
} dataset_t;

typedef struct neighbors_t {
   int neighbor_count;
   int neighbor[ OBSERVATIONS ];
} neighbors_t;

#define SQR( x )    ( ( x ) * ( x ) )

//#define EPSILON        1.5
//#define MINPTS         5

#define EPSILON        1.70
#define MINPTS         4

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

double distance( int i, int j )
{
   double sum = 0.0;

   for ( int feature = 0 ; feature < FEATURES ; feature++ )
   {
      sum += SQR( ( dataset[ i ].features[ feature ] - dataset[ j ].features[ feature ] ) );
   }

   return sqrt( sum );
}


neighbors_t *find_neighbors( int observation )
{
   neighbors_t *neighbor = ( neighbors_t * )malloc( sizeof( neighbors_t ) );

   bzero( (void *)neighbor, sizeof( neighbors_t ) );

   for ( int i = 0 ; i < OBSERVATIONS ; i++ )
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
   free( neighbors );

   return;
}

void fold_neighbors( neighbors_t *seed_set, neighbors_t *neighbors )
{
   for ( int i = 0 ; i < OBSERVATIONS ; i++ )
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
   for ( int i = 0 ; i < OBSERVATIONS ; i++ )
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

   for ( int i = 0 ; i < OBSERVATIONS ; i++ )
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

void print_dataset(){
   for(int i=0; i< OBSERVATIONS; i++){
      printf("%d, %s, ", i, dataset[i].name);
      
      for(int j=0; j< FEATURES; j++){
         printf("%d, ", dataset[i].features[j]);
      }

      printf("%d, ", dataset[i].class);
      printf("%d", dataset[i].label);
      printf("\n");
   }

}

void load_dataset(){
   FILE *fp;
   char buffer[10000];
   char *pbuff;
   char *ch;
   char delim[] = ",";

   dataset = (dataset_t *) malloc(sizeof(dataset_t)*OBSERVATIONS);

   fp = fopen("./dataset.csv", "r");

   int struct_counter = 0;
   int observation_count = 0;

   while (1) {
      if (!fgets(buffer, sizeof buffer, fp)) break;
      pbuff = buffer;

      ch = strtok(pbuff, delim);

      while (ch != NULL) {
         if(struct_counter == 0){
            dataset[observation_count].name = (char *)malloc(sizeof(char)*strlen(ch));
            strcpy(dataset[observation_count].name, ch);
         }
         else if(struct_counter>=1 && struct_counter <= FEATURES){
            dataset[observation_count].features[struct_counter - 1] = atoi(ch);
         }
         else if(struct_counter == FEATURES + 1){
            dataset[observation_count].class = atoi(ch);
         }
         else if(struct_counter == FEATURES + 2){
            dataset[observation_count].label = atoi(ch);
         }
         else{
            assert(struct_counter <= FEATURES + 2);
         }

         ch = strtok(NULL, delim);
         struct_counter++;
      }

      observation_count++;
      struct_counter = 0;
   }
   
   fclose(fp);

}

void free_dataset(){
   for(int i=0; i< OBSERVATIONS; i++){
      free(dataset[i].name);
   }

   free(dataset);

}


void emit_classes(int clusters){
   for ( int class = 1 ; class <= clusters ; class++ )
   {
      printf( "Class %d:\n", class );
      for ( int obs = 0 ; obs < OBSERVATIONS ; obs++ )
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
   for ( int obs = 0 ; obs < OBSERVATIONS ; obs++ )
   {
      if ( dataset[ obs ].label == NOISE )
      {
         printf("  %s (%d)\n", dataset[ obs ].name, dataset[ obs ].class );
      }
   }
   printf("\n");
}


int main( void )
{
   int clusters;

   load_dataset();

   clusters = dbscan( );

   // emit classes
   emit_classes(clusters);

   // Emit outliers (NOISE)
   emit_outliers();

   free_dataset();

   return 0;

}
