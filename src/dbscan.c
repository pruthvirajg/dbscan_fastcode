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
#include "../include/acc_distance.h"

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


float ref_distance(DTYPE_OBS i, DTYPE_OBS j )
{
   float sum = 0.0;
   ref_dst_call_count += 1;

   ref_dst_st = rdtsc();
   for ( int feature = 0 ; feature < FEATURES ; feature++ )
   {
      sum += SQR( ( dataset[ i ].features[ feature ] - dataset[ j ].features[ feature ] ) );
   }
   ref_dst_et = rdtsc();
   
   ref_dst_cycles += (ref_dst_et - ref_dst_st);

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
   int correct_min_pts;
   int correct_eps_mat;
   /***
    * Re-written schedule for DBSCAN to support acceleration
   */

   // Calculate the distance and generate epsilon matrix
   // TODO: Replace this with accelerated distance calc kernel
   #ifdef VERIFY_ACC
   gen_epsilon_matrix();
   acc_distance_simd();
   correct_eps_mat = verify_eps_mat();
   assert(correct_eps_mat==1);
   #else
   acc_dst_st = rdtsc();
   
   acc_distance_simd();

   acc_dst_et = rdtsc();   
   acc_dst_cycles += (acc_dst_et - acc_dst_st);

   #endif

   #ifdef DUMP_EPSILON_MAT
   FILE *fp;
   FILE *ref_fp;
   fp = fopen("./epsilon_matrix.csv", "w");
   ref_fp = fopen("./ref_epsilon_matrix.csv", "w");
   char buffer[10000];
   char template[] = "%d,";

   char ref_buffer[10000];
   char ref_template[] = "%d,";

   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      for (DTYPE_OBS j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         sprintf(buffer, template, epsilon_matrix[i*TOTAL_OBSERVATIONS + j]);
         fputs(buffer, fp);

         sprintf(ref_buffer, ref_template, ref_epsilon_matrix[i*TOTAL_OBSERVATIONS + j]);
         fputs(ref_buffer, ref_fp);
      }
      fputs("\n", fp);
      fputs("\n", ref_fp);
   }

   fclose(fp);
   fclose(ref_fp);
   return 0;
   #endif

   // Reduction along the rows to check if row has > MIN_PTS
   #ifdef VERIFY_ACC
   calc_min_pts();
   acc_min_pts();
   correct_min_pts = verify_min_pts();
   assert(correct_min_pts==0);
   #else
   min_pts_st = rdtsc();

   acc_min_pts();
   
   min_pts_et = rdtsc();   
   min_pts_cycles += (min_pts_et - min_pts_st);
   #endif
   
   // Label all points
   clusters = class_label();
   return clusters;
}


bool acc_distance(DTYPE_OBS i, DTYPE_OBS j )
{
   // reference implementation for SIMD distance
   float distance = 0.0;
   acc_dst_call_count += 1;
   int res = 0;

   acc_dst_st = rdtsc();
   for ( int feature = 0 ; feature < FEATURES ; feature++ )
   {
      distance += SQR( ( dataset[ i ].features[ feature ] - dataset[ j ].features[ feature ] ) );
   }
   acc_dst_et = rdtsc();
   
   acc_dst_cycles += (acc_dst_et - acc_dst_st);

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
         if(i!=j) ref_epsilon_matrix[i*TOTAL_OBSERVATIONS + j] = acc_distance(i, j);
         
      }
   }
}

int verify_eps_mat(void){
   int correct = 1;
   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      for (DTYPE_OBS j = 0 ; j < TOTAL_OBSERVATIONS ; j++ ){
         correct &= (ref_epsilon_matrix[i*TOTAL_OBSERVATIONS + j] == epsilon_matrix[i*TOTAL_OBSERVATIONS + j]);
      }
   }
   return correct;
}

void acc_min_pts(void){
   // Accelerate min points using popcnt
   // Options:
   // __int64 _mm_popcnt_u64 (unsigned __int64 a) 
   // int _mm_popcnt_u32 (unsigned int a)
   // latency 3, throughput 1

   __uint64_t num_valid_points_0 = 0;
   __uint64_t query_0;

   __uint64_t num_valid_points_1 = 0;
   __uint64_t query_1;

   __uint64_t num_valid_points_2 = 0;
   __uint64_t query_2;

   __uint64_t num_valid_points_3 = 0;
   __uint64_t query_3;

   __uint64_t num_valid_points_4 = 0;
   __uint64_t query_4;

   __uint64_t num_valid_points_5 = 0;
   __uint64_t query_5;

   __uint64_t res_0, res_1, res_2, res_3, res_4, res_5;

   bool *eps_mat_ptr_0 = epsilon_matrix;
   bool *eps_mat_ptr_1 = epsilon_matrix;
   bool *eps_mat_ptr_2 = epsilon_matrix;
   bool *eps_mat_ptr_3 = epsilon_matrix;
   bool *eps_mat_ptr_4 = epsilon_matrix;
   bool *eps_mat_ptr_5 = epsilon_matrix;

   // Reduction along the rows to check if row has > MIN_PTS
   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i+=6 ){
      // For each row check if MIN_PTS is met
      num_valid_points_0 = 0;
      num_valid_points_1 = 0;
      num_valid_points_2 = 0;
      num_valid_points_3 = 0;
      num_valid_points_4 = 0;
      num_valid_points_5 = 0;

      // row stride
      eps_mat_ptr_0 = epsilon_matrix + i * TOTAL_OBSERVATIONS;
      eps_mat_ptr_1 = epsilon_matrix + (i+1) * TOTAL_OBSERVATIONS;
      eps_mat_ptr_2 = epsilon_matrix + (i+2) * TOTAL_OBSERVATIONS;
      eps_mat_ptr_3 = epsilon_matrix + (i+3) * TOTAL_OBSERVATIONS;
      eps_mat_ptr_4 = epsilon_matrix + (i+4) * TOTAL_OBSERVATIONS;
      eps_mat_ptr_5 = epsilon_matrix + (i+5) * TOTAL_OBSERVATIONS;

      // min_pts_st = rdtsc();
   
      for (DTYPE_OBS j = 0 ; j < TOTAL_OBSERVATIONS/8 ; j++ ){
         // col stride
         // query_0 = *(((__uint64_t*)eps_mat_ptr_0) + j);
         // query_1 = *(((__uint64_t*)eps_mat_ptr_1) + j);
         // query_2 = *(((__uint64_t*)eps_mat_ptr_2) + j);
         // query_3 = *(((__uint64_t*)eps_mat_ptr_3) + j);
         // query_4 = *(((__uint64_t*)eps_mat_ptr_4) + j);
         // query_5 = *(((__uint64_t*)eps_mat_ptr_5) + j);
         
         
         // res_0 = _mm_popcnt_u64(query_0);
         // res_1 = _mm_popcnt_u64(query_1);
         // res_2 = _mm_popcnt_u64(query_2);
         // res_3 = _mm_popcnt_u64(query_3);
         // res_4 = _mm_popcnt_u64(query_4);
         // res_5 = _mm_popcnt_u64(query_5);

         // num_valid_points_0 += res_0;
         // num_valid_points_1 += res_1;
         // num_valid_points_2 += res_2;
         // num_valid_points_3 += res_3;
         // num_valid_points_4 += res_4;
         // num_valid_points_5 += res_5;
         query_0 = *(((__uint64_t*)eps_mat_ptr_0) + j);
         res_0 = _mm_popcnt_u64(query_0);
         num_valid_points_0 += res_0;

         query_1 = *(((__uint64_t*)eps_mat_ptr_1) + j);
         res_1 = _mm_popcnt_u64(query_1);
         num_valid_points_1 += res_1;

         query_2 = *(((__uint64_t*)eps_mat_ptr_2) + j);
         res_2 = _mm_popcnt_u64(query_2);
         num_valid_points_2 += res_2;

         query_3 = *(((__uint64_t*)eps_mat_ptr_3) + j);
         res_3 = _mm_popcnt_u64(query_3);
         num_valid_points_3 += res_3;

         query_4 = *(((__uint64_t*)eps_mat_ptr_4) + j);
         res_4 = _mm_popcnt_u64(query_4);
         num_valid_points_4 += res_4;

         query_5 = *(((__uint64_t*)eps_mat_ptr_5) + j);
         res_5 = _mm_popcnt_u64(query_5);
         num_valid_points_5 += res_5;
      }

      // min_pts_et = rdtsc();   
      // min_pts_cycles += (min_pts_et - min_pts_st);

      min_pts_vector[i] = (num_valid_points_0 >= MINPTS) ? true: false;
      min_pts_vector[i+1] = (num_valid_points_1 >= MINPTS) ? true: false;
      min_pts_vector[i+2] = (num_valid_points_2 >= MINPTS) ? true: false;
      min_pts_vector[i+3] = (num_valid_points_3 >= MINPTS) ? true: false;
      min_pts_vector[i+4] = (num_valid_points_4 >= MINPTS) ? true: false;
      min_pts_vector[i+5] = (num_valid_points_5 >= MINPTS) ? true: false;
   }
}


void calc_min_pts(void){
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

      ref_min_pts_vector[i] = (num_valid_points >= MINPTS) ? true: false;
   }
}


int verify_min_pts(void){
   int correct = 1;
   for (DTYPE_OBS i = 0 ; i < TOTAL_OBSERVATIONS ; i++ ){
      correct &= (min_pts_vector[i] != ref_min_pts_vector[i]);
   }
   return correct;
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
      traverse_mask[i] = 1;
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

      // Assign neighbour to the cluster
      if(epsilon_matrix[row_index*TOTAL_OBSERVATIONS + j]){
         dataset[j].label = core_pt_label;
      

         if(!traverse_mask[j]){
            // if not yet traversed, add row_index to N_List
            queue_insert_tail(N_List, j);

            // set traverse_mask
            traverse_mask[j] = 1;
         }
      }
   }

   // process neighbours to label density reachable points
   while(queue_size(N_List) != 0){
      traverse_row(N_List->head->row_index, cluster, core_pt_label);
      queue_remove_head(N_List);
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

