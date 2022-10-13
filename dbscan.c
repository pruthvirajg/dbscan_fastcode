#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <x86intrin.h>
#include <immintrin.h>

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#define MAX_FREQ 3.2
#define BASE_FREQ 2.4
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
   float  features[ FEATURES ];
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

dataset_t dataset[ OBSERVATIONS ] = 
{
//  Name        Features                           Class
   {"aardvark", {1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1}, 1, 0},
   {"antelope", {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"bass",     {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"bear",     {1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1}, 1, 0},
   {"boar",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"buffalo",  {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"calf",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1}, 1, 0},
   {"carp",     {0,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0}, 4, 0},
   {"catfish",  {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"cavy",     {1,0,0,1,0,0,0,1,1,1,0,0,4,0,1,0}, 1, 0},
   {"cheetah",  {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"chicken",  {0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0}, 2, 0},
   {"chub",     {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"crab",     {0,0,1,0,0,1,1,0,0,0,0,0,4,0,0,0}, 7, 0},
   {"crayfish", {0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0}, 7, 0},
   {"crow",     {0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"deer",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"dogfish",  {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1}, 4, 0},
   {"dolphin",  {0,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1}, 1, 0},
   {"dove",     {0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0}, 2, 0},
   {"duck",     {0,1,1,0,1,1,0,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"elephant", {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"flamingo", {0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,1}, 2, 0},
   {"flea",     {0,0,1,0,0,0,0,0,0,1,0,0,6,0,0,0}, 6, 0},
   {"frog",     {0,0,1,0,0,1,1,1,1,1,0,0,4,0,0,0}, 5, 0},
   {"frog",     {0,0,1,0,0,1,1,1,1,1,1,0,4,0,0,0}, 5, 0},
   {"fruitbat", {1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0}, 1, 0},
   {"giraffe",  {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"girl",     {1,0,0,1,0,0,1,1,1,1,0,0,2,0,1,1}, 1, 0},
   {"gnat",     {0,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0}, 6, 0},
   {"goat",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1}, 1, 0},
   {"gorilla",  {1,0,0,1,0,0,0,1,1,1,0,0,2,0,0,1}, 1, 0},
   {"gull",     {0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"haddock",  {0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"hamster",  {1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,0}, 1, 0},
   {"hare",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,0}, 1, 0},
   {"hawk",     {0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"herring",  {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"honeybee", {1,0,1,0,1,0,0,0,0,1,1,0,6,0,1,0}, 6, 0},
   {"housefly", {1,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0}, 6, 0},
   {"kiwi",     {0,1,1,0,0,0,1,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"ladybird", {0,0,1,0,1,0,1,0,0,1,0,0,6,0,0,0}, 6, 0},
   {"lark",     {0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"leopard",  {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"lion",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"lobster",  {0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0}, 7, 0},
   {"lynx",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"mink",     {1,0,0,1,0,1,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"mole",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,0}, 1, 0},
   {"mongoose", {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"moth",     {1,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0}, 6, 0},
   {"newt",     {0,0,1,0,0,1,1,1,1,1,0,0,4,1,0,0}, 5, 0},
   {"octopus",  {0,0,1,0,0,1,1,0,0,0,0,0,8,0,0,1}, 7, 0},
   {"opossum",  {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,0}, 1, 0},
   {"oryx",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"ostrich",  {0,1,1,0,0,0,0,0,1,1,0,0,2,1,0,1}, 2, 0},
   {"parakeet", {0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0}, 2, 0},
   {"penguin",  {0,1,1,0,0,1,1,0,1,1,0,0,2,1,0,1}, 2, 0},
   {"pheasant", {0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"pike",     {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1}, 4, 0},
   {"piranha",  {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"pitviper", {0,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0}, 3, 0},
   {"platypus", {1,0,1,1,0,1,1,0,1,1,0,0,4,1,0,1}, 1, 0},
   {"polecat",  {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"pony",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1}, 1, 0},
   {"porpoise", {0,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1}, 1, 0},
   {"puma",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"pussycat", {1,0,0,1,0,0,1,1,1,1,0,0,4,1,1,1}, 1, 0},
   {"raccoon",  {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"reindeer", {1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1}, 1, 0},
   {"rhea",     {0,1,1,0,0,0,1,0,1,1,0,0,2,1,0,1}, 2, 0},
   {"scorpion", {0,0,0,0,0,0,1,0,0,1,1,0,8,1,0,0}, 7, 0},
   {"seahorse", {0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"seal",     {1,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1}, 1, 0},
   {"sealion",  {1,0,0,1,0,1,1,1,1,1,0,1,2,1,0,1}, 1, 0},
   {"seasnake", {0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0}, 3, 0},
   {"seawasp",  {0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0}, 7, 0},
   {"skimmer",  {0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"skua",     {0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"slowworm", {0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0}, 3, 0},
   {"slug",     {0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0}, 7, 0},
   {"sole",     {0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"sparrow",  {0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0}, 2, 0},
   {"squirrel", {1,0,0,1,0,0,0,1,1,1,0,0,2,1,0,0}, 1, 0},
   {"starfish", {0,0,1,0,0,1,1,0,0,0,0,0,5,0,0,0}, 7, 0},
   {"stingray", {0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1}, 4, 0},
   {"swan",     {0,1,1,0,1,1,0,0,1,1,0,0,2,1,0,1}, 2, 0},
   {"termite",  {0,0,1,0,0,0,0,0,0,1,0,0,6,0,0,0}, 6, 0},
   {"toad",     {0,0,1,0,0,1,0,1,1,1,0,0,4,0,0,0}, 5, 0},
   {"tortoise", {0,0,1,0,0,0,0,0,1,1,0,0,4,1,0,1}, 3, 0},
   {"tuatara",  {0,0,1,0,0,0,1,1,1,1,0,0,4,1,0,0}, 3, 0},
   {"tuna",     {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1}, 4, 0},
   {"vampire",  {1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0}, 1, 0},
   {"vole",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,0}, 1, 0},
   {"vulture",  {0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,1}, 2, 0},
   {"wallaby",  {1,0,0,1,0,0,0,1,1,1,0,0,2,1,0,1}, 1, 0},
   {"wasp",     {1,0,1,0,1,0,0,0,0,1,1,0,6,0,0,0}, 6, 0},
   {"wolf",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"worm",     {0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0}, 7, 0},
   {"wren",     {0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0}, 2, 0},

   {"flea with teeth", 
                {0,0,1,0,0,0,0,1,0,1,0,0,6,0,0,0}, 0, 0},
   {"predator with hair", 
                {0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, 0, 0},
   {"six legged aquatic egg layer", 
                {0,0,1,0,0,1,0,0,0,0,0,0,6,0,0,0}, 0, 0},

};

// Globals for distance performance
int dst_call_count = 0;
unsigned long long dst_st = 0;
unsigned long long dst_et = 0;
unsigned long long dst_cycles = 0;


double distance( int i, int j )
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


neighbors_t *find_neighbors( int observation )
{
   neighbors_t *neighbor = ( neighbors_t * )malloc( sizeof( neighbors_t ) );

   bzero( (void *)neighbor, sizeof( neighbors_t ) );

   for ( int i = 0 ; i < OBSERVATIONS ; i++ )
   {
      if ( i == observation ) continue;

      if ( distance( observation, i ) <= EPSILON ) //dbscan calls find neighbors
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
         // NOT NOISE
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


int main( void )
{
   unsigned long long cycles = 0;
   unsigned long long st = 0;
   unsigned long long et = 0;
   double dst_percentage = 0;

   int clusters;

   // For N runs
   unsigned long long runs = 100;
   for(unsigned long long i = 0; i < runs; i++){

      st = rdtsc();
      clusters = dbscan( );
      et = rdtsc();
      cycles += (et-st);
      // Reset labels for all runs and skip for the last 
      for( int j = 0; j < OBSERVATIONS; j++){
         if(i == (runs-1)){
            break;
         }
         dataset[j].label = UNDEFINED;
      }
   }
   dst_percentage = ( (double)dst_cycles / (double)cycles ) * 100;

   // emit classes
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

   // Emit outliers (NOISE)
   printf( "NOISE\n" );
   for ( int obs = 0 ; obs < OBSERVATIONS ; obs++ )
   {
      if ( dataset[ obs ].label == NOISE )
      {
         printf("  %s (%d)\n", dataset[ obs ].name, dataset[ obs ].class );
      }
   }
   printf("\n");

   // Dataset
   printf("Dataset Metrics:\n");
   printf("Number of datapoints: %d, Number of features: %d\n\n", OBSERVATIONS, FEATURES);

   // For N runs
   printf("Performance Metrics Over N = %llu Runs:\n",runs);
   printf("RDTSC Base Cycles Taken for dbscan: %llu\n", cycles);
   printf("RDTSC Base Cycles Taken for distance: %llu\n", dst_cycles);

   printf("TURBO Cycles Taken for dbscan: %f\n", cycles * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for distance: %f\n", dst_cycles * ((double)MAX_FREQ)/BASE_FREQ);

   printf("Percentage of Cycles Spent Calculating distance: %% %f\n", dst_percentage);
   printf("Distance is called %d times, each of which takes ~%f cycles\n", dst_call_count, ((double) dst_cycles / (double)dst_call_count));

   // Average Performance
   printf("\nAverage Performance Metrics:\n");

   printf("Average RDTSC Base Cycles Taken for dbscan: %llu\n", (cycles/runs));
   printf("Average RDTSC Base Cycles Taken for distance: %llu\n", (dst_cycles/runs));

   printf("TURBO Cycles Taken for dbscan: %f\n", (cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for distance: %f\n", (dst_cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);

   printf("Percentage of Cycles Spent Calculating distance: %% %f\n", dst_percentage);
   printf("Distance is called %llu times, each of which takes ~%f cycles\n", (dst_call_count / runs), ((double) dst_cycles / (double)dst_call_count));

   // Peak Performance
   printf("\nPeak Performance:\n");
   printf("Peak FLOPS/Cycle = 24.00\n");
   printf("Peak GFLOPS/Second = 76.80\n");

   // Baseline Performance 
   printf("\nBaseline Performance:\n");
   printf("Number of operations peformed in each run of DBSCAN is %llu\n", (dst_call_count/runs)*3);
   printf("Total number of cycles spent in the core distance function is %llu\n", (dst_cycles/runs));
   printf("Baseline FLOPS/Cycle = %f\n", (dst_call_count*3.0) / (dst_cycles));
   printf("Baseline GFLOPS/Second = %f", ((dst_call_count*3.0) / (dst_cycles)) * MAX_FREQ );
 
   return 0;
}
