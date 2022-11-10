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

int main(int argc, char** argv)
{
   unsigned long long cycles = 0;
   unsigned long long st = 0;
   unsigned long long et = 0;
   double dst_percentage = 0;
   double min_pts_percentage = 0;

   dst_call_count = 0;
   dst_st = 0;
   dst_et  = 0;
   dst_cycles = 0;

   min_pts_st = 0;
   min_pts_et  = 0;
   min_pts_cycles = 0;

   int clusters;

   // get cmd line arg to run ref-dbscan or acc-dbscan
   if(argc == 2 && atoi(argv[1]) == 1){
      ACC_DBSCAN = true;
      printf("Running acc_dbscan()\n");
   }
   else{
      ACC_DBSCAN = false;
      printf("Running ref_dbscan() \n");
   }


   EPSILON_SQUARE = EPSILON * EPSILON;

   load_dataset();
   
   // allocate memory for epsilon_matrix
   #ifdef VERIFY_ACC
   ref_epsilon_matrix = (bool *) calloc(TOTAL_OBSERVATIONS * TOTAL_OBSERVATIONS, sizeof(bool));
   if (ref_epsilon_matrix == NULL) {
         printf("ref_epsilon_matrix memory not allocated.\n");
         exit(0);
   }
   #endif

   epsilon_matrix = (bool *) calloc(TOTAL_OBSERVATIONS * TOTAL_OBSERVATIONS, sizeof(bool));
   if (epsilon_matrix == NULL) {
         printf("epsilon_matrix memory not allocated.\n");
         exit(0);
   }
   
   // allocate memory for min_pts_vector
   #ifdef VERIFY_ACC
   ref_min_pts_vector = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));
   if (ref_min_pts_vector == NULL) {
         printf("ref_min_pts_vector memory not allocated.\n");
         exit(0);
   }
   #endif

   min_pts_vector = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));
   if (min_pts_vector == NULL) {
         printf("min_pts_vector memory not allocated.\n");
         exit(0);
   }

   // allocate memory for class label row traverse_mask
   traverse_mask = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));

   // Profile for N runs
   unsigned long long runs = (unsigned long long) NUM_RUNS;

   for(unsigned long long i = 0; i < runs; i++){

      st = rdtsc();
      clusters = ACC_DBSCAN ? acc_dbscan() : ref_dbscan( );

      et = rdtsc();

      cycles += (et-st);
      
      // Reset labels for all runs and skip for the last 
      for( int j = 0; j < TOTAL_OBSERVATIONS; j++){
         if(i == (runs-1)){
            break;
         }

         
         dataset[j].label = ACC_DBSCAN ? NOISE : UNDEFINED;
         
      }

      free(epsilon_matrix);
      
      #ifdef VERIFY_ACC
      free(ref_min_pts_vector);
      free(ref_epsilon_matrix);
      #endif

      free(min_pts_vector);

      epsilon_matrix = (bool *) calloc(TOTAL_OBSERVATIONS * TOTAL_OBSERVATIONS, sizeof(bool));
      #ifdef VERIFY_ACC
      ref_min_pts_vector = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));
      ref_epsilon_matrix = (bool *) calloc(TOTAL_OBSERVATIONS * TOTAL_OBSERVATIONS, sizeof(bool));
      #endif
      min_pts_vector = (bool *) calloc(TOTAL_OBSERVATIONS, sizeof(bool));
   }

   dst_percentage = ( (double)dst_cycles / (double)cycles ) * 100;

   min_pts_percentage = ( (double)min_pts_cycles / (double)cycles ) * 100;

   // emit classes
   emit_classes(clusters);

   // Emit outliers (NOISE)
   emit_outliers();

   // free all memory
   free_dataset();
   free(epsilon_matrix);

   // Dataset
   unsigned long long dist_num_ops = runs * 3 * (TOTAL_OBSERVATIONS * (TOTAL_OBSERVATIONS - 1) / 2);

   unsigned long long min_pts_num_ops =  runs * pow(TOTAL_OBSERVATIONS, 2) * sizeof(__uint64_t) / sizeof(bool);

   printf("Dataset Metrics:\n");
   printf("Number of datapoints: %llu, Number of features: %d\n\n", TOTAL_OBSERVATIONS, FEATURES);

   // For N runs
   printf("Performance Metrics Over N = %llu Runs:\n", runs);
   printf("RDTSC Base Cycles Taken for dbscan: %llu\n", cycles);
   printf("RDTSC Base Cycles Taken for distance: %llu\n", dst_cycles);
   printf("RDTSC Base Cycles Taken for min pts: %llu\n", min_pts_cycles);

   printf("TURBO Cycles Taken for dbscan: %f\n", cycles * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for distance: %f\n", dst_cycles * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for min pts: %f\n", min_pts_cycles * ((double)MAX_FREQ)/BASE_FREQ);

   printf("Percentage of Cycles Spent Calculating distance: %% %f\n", dst_percentage);
   printf("Distance total number of ops: %llu, each of which takes ~%f cycles\n", dist_num_ops, ((double) dst_cycles / (double)dist_num_ops));

   printf("Percentage of Cycles Spent Calculating min pts: %% %f\n", min_pts_percentage);

   // Average Performance
   printf("\nAverage Performance Metrics:\n");

   printf("Average RDTSC Base Cycles Taken for dbscan: %llu\n", (cycles/runs));
   printf("Average RDTSC Base Cycles Taken for distance: %llu\n", (dst_cycles/runs));
   printf("Average RDTSC Base Cycles Taken for min pts: %llu\n", (min_pts_cycles/runs));

   printf("TURBO Cycles Taken for dbscan: %f\n", (cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for distance: %f\n", (dst_cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for min pts: %f\n", (min_pts_cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);

   printf("Percentage of Cycles Spent Calculating distance: %% %f\n", dst_percentage);
   printf("Distance number of ops/ run: %llu, each of which takes ~%f cycles\n", (dist_num_ops / runs), ((double) dst_cycles / (double)dist_num_ops));

   // Peak Performance Distance
   printf("\nPeak Performance Distance:\n");
   printf("Peak FLOPS/Cycle = 24.00\n");
   printf("Peak GFLOPS/Second = 76.80\n");

   // Baseline Performance Distance
   printf("\nAccerlated Distance Performance:\n");
   printf("Number of operations peformed in each run of DBSCAN is %llu\n", (dist_num_ops/runs));
   printf("Total number of cycles spent in the core distance function is %llu\n", (dst_cycles/runs));
   printf("Baseline FLOPS/Cycle = %f\n", ((double) dist_num_ops) / ((double) dst_cycles));
   printf("Baseline GFLOPS/Second = %f\n", (((double) dist_num_ops) / ((double) dst_cycles)) * MAX_FREQ );

   // Peak Performance Min Pts
   printf("\nPeak Performance Min Pts:\n");
   printf("Peak FLOPS/Cycle = 1.00\n");
   printf("Peak GFLOPS/Second = 3.20\n");

   // Baseline Performance Min Pts
   printf("\nAccerlated Min Pts Performance:\n");
   printf("Number of operations peformed in each run of DBSCAN is %llu\n", (min_pts_num_ops/runs));
   printf("Total number of cycles spent in the core min pts function is %llu\n", (min_pts_cycles/runs));

   if(min_pts_cycles > 0){
      printf("Baseline FLOPS/Cycle = %f\n", ((double) min_pts_num_ops) / ((double) min_pts_cycles));
      printf("Baseline GFLOPS/Second = %f\n", (((double) min_pts_num_ops) / ((double) min_pts_cycles)) * MAX_FREQ );
   }

   return 0;

}