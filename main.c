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
   
   double ref_dst_percentage = 0;
   double acc_dst_percentage = 0;
   double simd_dst_percentage = 0;

   double min_pts_percentage = 0;

   ref_dst_call_count = 0;
   ref_dst_st = 0;
   ref_dst_et  = 0;
   ref_dst_cycles = 0;

   acc_dst_call_count = 0;
   acc_dst_st = 0;
   acc_dst_et  = 0;
   acc_dst_cycles = 0;

   simd_dst_call_count = 0;
   simd_dst_st = 0;
   simd_dst_et  = 0;
   simd_dst_cycles = 0;

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

   acc_dst_percentage = ( (double)acc_dst_cycles / (double)cycles ) * 100;

   simd_dst_percentage = ( (double)simd_dst_cycles / (double)acc_dst_cycles) * 100;

   min_pts_percentage = ( (double)min_pts_cycles / (double)cycles ) * 100;

   // emit classes
   emit_classes(clusters);

   // Emit outliers (NOISE)
   emit_outliers();

   // free all memory
   free_dataset();
   free(epsilon_matrix);

   // Dataset
   // approx ops for upper triangular
   unsigned long long acc_dist_num_ops = runs * 3 * (TOTAL_OBSERVATIONS * (TOTAL_OBSERVATIONS - 1) / 2);
   
   // TODO: Change this to match exact number of SIMD ops done
   unsigned long long ops_sub = 6;
   unsigned long long ops_fmadd = 12;
   unsigned long long ops_cmp = 6;

   // unsigned long long simd_dist_num_ops = \
   //       simd_dst_call_count * (ops_sub + ops_fmadd) + \
   //       (simd_dst_call_count/ FEATURES) * ops_cmp;
   unsigned long long simd_dist_num_ops = 8 * simd_dst_call_count * (ops_sub + ops_fmadd);

   unsigned long long min_pts_num_ops =  runs * 3 * 6 * ((double)pow(TOTAL_OBSERVATIONS, 2) / (double)(6*8));

   printf("Dataset Metrics:\n");
   printf("Number of datapoints: %llu, Number of features: %d\n\n", TOTAL_OBSERVATIONS, FEATURES);

   // For N runs
   printf("Performance Metrics Over N = %llu Runs:\n", runs);
   printf("RDTSC Base Cycles Taken for dbscan: %llu\n", cycles);
   printf("RDTSC Base Cycles Taken for acc_distance: %llu\n", acc_dst_cycles);
   printf("RDTSC Base Cycles Taken for simd_distance: %llu\n", simd_dst_cycles);
   printf("RDTSC Base Cycles Taken for min pts: %llu\n", min_pts_cycles);

   printf("TURBO Cycles Taken for dbscan: %f\n", cycles * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for acc_distance: %f\n", acc_dst_cycles * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for simd_distance: %f\n", simd_dst_cycles * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for min pts: %f\n", min_pts_cycles * ((double)MAX_FREQ)/BASE_FREQ);

   printf("Percentage of Cycles Spent Calculating acc_distance: %% %f\n", acc_dst_percentage);
   printf("Percentage of Cycles Spent Calculating simd_distance: %% %f\n", simd_dst_percentage);
   printf("acc_Distance total number of ops: %llu, each of which takes ~%f cycles\n", acc_dist_num_ops, ((double) acc_dst_cycles / (double)acc_dist_num_ops));
   printf("simd_Distance total number of ops: %llu, each of which takes ~%f cycles\n", simd_dist_num_ops, ((double) simd_dst_cycles / (double)simd_dist_num_ops));

   printf("Percentage of Cycles Spent Calculating min pts: %% %f\n", min_pts_percentage);

   // Average Performance
   printf("\nAverage Performance Metrics:\n");

   printf("Average RDTSC Base Cycles Taken for dbscan: %llu\n", (cycles/runs));
   printf("Average RDTSC Base Cycles Taken for acc_distance: %llu\n", (acc_dst_cycles/runs));
   printf("Average RDTSC Base Cycles Taken for simd_distance: %llu\n", (simd_dst_cycles/runs));
   printf("Average RDTSC Base Cycles Taken for min pts: %llu\n", (min_pts_cycles/runs));

   printf("TURBO Cycles Taken for dbscan: %f\n", (cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for acc_distance: %f\n", (acc_dst_cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for simd_distance: %f\n", (simd_dst_cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);
   printf("TURBO Cycles Taken for min pts: %f\n", (min_pts_cycles/runs) * ((double)MAX_FREQ)/BASE_FREQ);

   printf("Percentage of Cycles Spent Calculating acc_distance: %% %f\n", acc_dst_percentage);
   printf("Percentage of Cycles Spent Calculating simd_distance: %% %f\n", simd_dst_percentage);
   printf("acc_Distance number of ops/ run: %llu, each of which takes ~%f cycles\n", (acc_dist_num_ops / runs), ((double)acc_dst_cycles / (double)acc_dist_num_ops));
   printf("simd_Distance number of ops/ run: %llu, each of which takes ~%f cycles\n", (simd_dist_num_ops / runs), ((double) simd_dst_cycles / (double)simd_dist_num_ops));

   // Peak Performance Distance
   printf("\nPeak Performance Distance:\n");
   printf("Peak FLOPS/Cycle = 24.00\n");
   printf("Peak GFLOPS/Second = 76.80\n");

   // Baseline Performance Distance
   printf("\nAccelerated Distance Performance:\n");
   printf("Number of operations peformed in each run of DBSCAN is %llu\n", (acc_dist_num_ops/runs));
   printf("Total number of cycles spent in the core distance function is %llu\n", (acc_dst_cycles/runs));
   printf("Baseline FLOPS/Cycle = %f\n", ((double) acc_dist_num_ops) / ((double) acc_dst_cycles));
   printf("Baseline GFLOPS/Second = %f\n", (((double) acc_dist_num_ops) / ((double) acc_dst_cycles)) * MAX_FREQ );

   printf("\nSIMD Distance Performance:\n");
   printf("Number of operations peformed in each run of DBSCAN is %llu\n", (simd_dist_num_ops/runs));
   printf("Total number of cycles spent in the core distance function is %llu\n", (simd_dst_cycles/runs));
   printf("Baseline FLOPS/Cycle = %f\n", ((double) simd_dist_num_ops) / ((double) simd_dst_cycles));
   printf("Baseline GFLOPS/Second = %f\n", (((double) simd_dist_num_ops) / ((double) simd_dst_cycles)) * MAX_FREQ );

   // Peak Performance Min Pts
   printf("\nPeak Performance Min Pts:\n");
   printf("Peak FLOPS/Cycle = 3.00\n");
   printf("Peak GFLOPS/Second = 9.60\n");

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