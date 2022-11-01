#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <stdbool.h>

#include "./include/dbscan.h"
#include "./include/utils.h"
#include "./include/config.h"


static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

// #define ACC_DBSCAN 1

int main( void )
{
   unsigned long long cycles = 0;
   unsigned long long st = 0;
   unsigned long long et = 0;
   double dst_percentage = 0;

   dst_call_count = 0;
   dst_st = 0;
   dst_et  = 0;
   dst_cycles = 0;

   int clusters;

   EPSILON_SQUARE = EPSILON * EPSILON;

   load_dataset();
   
   // allocate memory for epsilon_matrix
   epsilon_matrix = (bool *) calloc(TOTAL_OBSERVATIONS * TOTAL_OBSERVATIONS, sizeof(bool));
   if (epsilon_matrix == NULL) {
         printf("epsilon_matrix memory not allocated.\n");
         exit(0);
   }
   
   min_pts_vector = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));
   if (min_pts_vector == NULL) {
         printf("min_pts_vector memory not allocated.\n");
         exit(0);
   }

   // Profile for N runs
   unsigned long long runs = (unsigned long long) NUM_RUNS;

   for(unsigned long long i = 0; i < runs; i++){

      st = rdtsc();
      #ifdef ACC_DBSCAN
         clusters = acc_dbscan();
      #else
         clusters = ref_dbscan( );
      #endif

      et = rdtsc();

      cycles += (et-st);
      
      // Reset labels for all runs and skip for the last 
      for( int j = 0; j < TOTAL_OBSERVATIONS; j++){
         if(i == (runs-1)){
            break;
         }

         #ifdef ACC_DBSCAN
            dataset[j].label = NOISE;
         #else
            dataset[j].label = UNDEFINED;
         #endif
      }

      free(epsilon_matrix);
      free(min_pts_vector);

      epsilon_matrix = (bool *) calloc(TOTAL_OBSERVATIONS * TOTAL_OBSERVATIONS, sizeof(bool));
      min_pts_vector = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));
   }

   dst_percentage = ( (double)dst_cycles / (double)cycles ) * 100;

   // emit classes
   emit_classes(clusters);

   // Emit outliers (NOISE)
   emit_outliers();

   // free all memory
   free_dataset();
   free(epsilon_matrix);

   // Dataset
   printf("Dataset Metrics:\n");
   printf("Number of datapoints: %llu, Number of features: %d\n\n", TOTAL_OBSERVATIONS, FEATURES);

   // For N runs
   printf("Performance Metrics Over N = %llu Runs:\n", runs);
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
   printf("Baseline GFLOPS/Second = %f\n", ((dst_call_count*3.0) / (dst_cycles)) * MAX_FREQ );

   return 0;

}