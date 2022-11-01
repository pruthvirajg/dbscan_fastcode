#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

#define OBSERVATIONS	112
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
   float features[ FEATURES ];
   int  class;
   int  label;
} dataset_t;

typedef struct neighbors_t {
   int neighbor_count;
   int neighbor[ OBSERVATIONS ];
} neighbors_t;

#define EPSILON        1.70
#define MINPTS         4

void pt_3_sequential(dataset_t *data, uint16_t *eps_mat, int x, float eps, int counter, float iterator, int mem_order);
int sequential(dataset_t *data, uint16_t *eps_mat, int x, int y, float eps, int counter, float iterator, int mem_order, int base_counter);
void print_helper(uint16_t *eps_mat);

int distance(dataset_t *data){

   __m256 E;
   __m256 x0,x1,x2;
   __m256 y0,y1;
   __m256 x0_y0, x0_y1, x1_y0, x1_y1;
   __m256 c0,c1,c2,c3,c4,c5;

   c0 = _mm256_setzero_ps();
   c1 = _mm256_setzero_ps();
   c2 = _mm256_setzero_ps();
   c3 = _mm256_setzero_ps();
   c4 = _mm256_setzero_ps();
   c5 = _mm256_setzero_ps();

   float eps = (EPSILON*EPSILON);
   int N = OBSERVATIONS;

   uint16_t *eps_mat;

   eps_mat = (uint16_t*)malloc( ( ( ( N*N / 2 ) - N ) / 8 ) * sizeof(uint16_t) );
   if (eps_mat == NULL) {
      printf("Memory not allocated.\n");
      exit(0);
   }
   
   int counter = 0;
   int base_counter = 0;
  
   // 1 SIMD REG (EPSILON)
   E = _mm256_set1_ps (eps);

   // Integers for SIMD routine
   uint16_t d_x0_y0, d_x0_y1, d_x1_y0, d_x1_y1, d_x2_y0, d_x2_y1;

   // Variables for serial routine

   float iterator = 0;
   int pts_cnt;
   int mem_order;

   // Iterate over 3 X elements
   for(int x = 0; x < N; x+=3){

      base_counter = counter;
      // Don't do anything for last element
      // NO?  happens actually THIS WON"T HAPPEN IF N DIVISBLE BY 3
      if(x == (N-1)){
         break;
      }
      printf("ITERATION: %d\n", x);

      // Iterator is the number of loop iterations the below loop will go through
      // Iterator should tell you how many times you do the SIMD portion 
      // Corner case at x = 93
      //iterator = floor((N-(x+3))/16);
      pts_cnt = ((N-1) - (x+3));
      iterator = floor(pts_cnt/16);
      printf("(N-1)-(x+3) = %d\n",(N-1)-(x+3) );
      printf("(N-1)-(x+3) %% 16 = %d\n",((N-1)-(x+3))%16 );
      printf("ITERATOR: %d\n",(int)iterator);

      // Decide Case for memory ordering
      // Case 1: SIMD
      // Case 2: Sequential
      // Case 3: SIMD, Sequential
      if( ((pts_cnt % 16) == 0) && (pts_cnt != 0)){
         mem_order = 1;
      }
      else if( (pts_cnt % 16) == pts_cnt){
         mem_order = 2;
      }
      else{
         mem_order = 3;
      }

      printf("CASE %d\n",mem_order);

      // Decide conditionally when to do this
      pt_3_sequential(data, eps_mat, x, eps, counter, iterator, mem_order);
      // if this the last, then break

      // NEEDS TO BE AN IF HERE AS WELL 
      for(int y = x+3; y < N; y+=16){

         // If you enter this without 16 points to compute, then don't do SIMD
         // Perform SIMD IFF we have at least 16 points to compute
         //if( (iterator != 0) && (y+16 < N) ){
         //if(y+16 < N){
         
         if( ((mem_order == 1) || (mem_order == 3)) && (y+16 < N) ){

            for(int ftr = 0; ftr < FEATURES; ftr++){
               //------------------------------- SIMD PORTION ---------------------------------

               // 3 SIMD REG (BROADCAST POINTS)
               x0 = _mm256_set1_ps (data[x].features[ftr]);
               x1 = _mm256_set1_ps (data[x+1].features[ftr]);
               x2 = _mm256_set1_ps (data[x+2].features[ftr]);

               // 2 SIMD REG (DISTANCE POINTS)
               y0 = _mm256_set_ps (data[y+7].features[ftr], data[y+6].features[ftr], data[y+5].features[ftr], data[y+4].features[ftr], data[y+3].features[ftr], data[y+2].features[ftr], data[y+1].features[ftr], data[y].features[ftr]);
               y1 = _mm256_set_ps (data[y+15].features[ftr], data[y+14].features[ftr], data[y+13].features[ftr], data[y+12].features[ftr], data[y+11].features[ftr], data[y+10].features[ftr], data[y+9].features[ftr], data[y+8].features[ftr]);

               // 4 SIMD REG (INTERMEDIATE SUB)
               x0_y0 = _mm256_sub_ps (x0, y0);
               x0_y1 = _mm256_sub_ps (x0, y1);
               x1_y0 = _mm256_sub_ps (x1, y0);
               x1_y1 = _mm256_sub_ps (x1, y1);
               y0 = _mm256_sub_ps (x2, y0);
               y1 = _mm256_sub_ps (x2, y1);

               // 6 SIMD REG (OUTPUT)
               c0 = _mm256_fmadd_ps (x0_y0, x0_y0, c0);
               c1 = _mm256_fmadd_ps (x0_y1, x0_y1, c1);
               c2 = _mm256_fmadd_ps (x1_y0, x1_y0, c2);
               c3 = _mm256_fmadd_ps (x1_y1, x1_y1, c3);
               c4 = _mm256_fmadd_ps (y0, y0, c4);
               c5 = _mm256_fmadd_ps (y1, y1, c5);
            }

               c0 = _mm256_cmp_ps (c0, E, _CMP_LE_OQ);
               c1 = _mm256_cmp_ps (c1, E, _CMP_LE_OQ);
               c2 = _mm256_cmp_ps (c2, E, _CMP_LE_OQ);
               c3 = _mm256_cmp_ps (c3, E, _CMP_LE_OQ);
               c4 = _mm256_cmp_ps (c4, E, _CMP_LE_OQ);
               c5 = _mm256_cmp_ps (c5, E, _CMP_LE_OQ);
               
               d_x0_y0 = _mm256_movemask_ps (c0);
               d_x0_y1 = _mm256_movemask_ps (c1);
               d_x1_y0 = _mm256_movemask_ps (c2);
               d_x1_y1 = _mm256_movemask_ps (c3);
               d_x2_y0 = _mm256_movemask_ps (c4);
               d_x2_y1 = _mm256_movemask_ps (c5);

               d_x0_y0 = ( (d_x0_y0 << 8 ) | d_x0_y1 );
               d_x1_y0 = ( (d_x1_y0 << 8 ) | d_x1_y1 );
               d_x2_y0 = ( (d_x2_y0 << 8 ) | d_x2_y1 );

               printf("\t\tSIMD PORTION WITH y = %d\n",y);
               printf("\t\t\t d_x0_y0 = %d,  d_x1_y0 = %d, d_x2_y0 = %d\n", d_x0_y0, d_x1_y0, d_x2_y0);

               // Decide how to store               

               eps_mat[counter + 1] = d_x0_y0;

               switch(mem_order)
               {
                  case 1:
                     eps_mat[counter + 1 + (int)iterator + 1] = d_x1_y0;
                     eps_mat[counter + 1 + (int)iterator + 1 + (int)iterator] = d_x2_y0;
                  case 3:
                     eps_mat[counter + 1 + (int)iterator + 1 + 1] = d_x1_y0;
                     eps_mat[counter + 1 + (int)iterator + 1 + 1 + (int)iterator + 1] = d_x2_y0;
               }
               counter++;
         
               // RESET C REG TO 0   
               c0 = _mm256_setzero_ps();
               c1 = _mm256_setzero_ps();
               c2 = _mm256_setzero_ps();
               c3 = _mm256_setzero_ps();
               c4 = _mm256_setzero_ps();
               c5 = _mm256_setzero_ps();
               //------------------------------- SIMD PORTION ---------------------------------
         }

         // If this is pure SIMD, and this is the last iteration, update counter to point to the start of the next block
         if( (mem_order == 1) && (y == N-1) ){
            counter = base_counter + 1 + (int)iterator + 1 + (int)iterator + (int)iterator;
         }

         // If this is SIMD followed by sequential, and this is the last iteration of SIMD,
         // The latest updated counter with other offset, will already point to the correct location
         // HAVE TO UPDATE COUNTER
         if( (mem_order == 3) && ((y+16) >= N)){
            //counter++;
            counter = sequential(data, eps_mat, x, y, eps, counter, iterator, mem_order, base_counter);
            break;
         }
      
         // has to be 2
         if (mem_order == 2){
            counter = sequential(data, eps_mat, x, y, eps, counter, iterator, mem_order, base_counter);
            break;
         }
      }
    }


 
   //  int nextline = 1;
   //  for(int l = 0; l < (( ( N*N / 2 ) - N ) / 8); l++){
   //    if(l == ( (N-nextline) / 8)){
   //      printf("\n");
   //      nextline++;
   //    }
   //    else{
   //      printf("%d  ",eps_mat[l]);
   //    }
   // }


   // for(int l = 0; l < (( ( N*N / 2 ) - N ) / 8); l++){
   //    printf("%d ",eps_mat[l]);
   // }
   printf("\n");
   print_helper(eps_mat);

   

   free(eps_mat);

   return 1;
}

void print_helper(uint16_t *eps_mat){

   int pts_cnt;
   int iterator;
   int N = OBSERVATIONS;
   int mem_order;
   int block_size;
   int base;
   int new_base = 0;

   //Naive
   for(int l = 0; l < 500; l++){
      printf("%d ",eps_mat[l]);
   }
   printf("\n\n");


   for(int x = 0; x < N; x+=3){
      //printf("x is %d\n",x);
      // This would only enter if N is a multiple of 3
      if(x == (N-1)){
         break;
      }
      pts_cnt = ((N-1) - (x+3));
      iterator = floor(pts_cnt/16);
      if( (pts_cnt % 16) == 0  && (pts_cnt != 0)){
         mem_order = 1;
      }
      else if( (pts_cnt % 16) == pts_cnt){
         mem_order = 2;
      }
      else{
         mem_order = 3;
      }

      // The base needs to change, it's not x.
      base = new_base;

      if(mem_order == 1){
         block_size = base + 2 + 3*(iterator);
         for(int l = base; l < block_size; l++){
            // New line after each section
            if(l == base + 1 + iterator){
               printf("\n");
            }
            else if(l == base +  2*(1+iterator)){
               printf("\n");
            }
            // 3 Sections
            if(l < 1 + iterator){
               printf("%d ", eps_mat[l]);
            }
            else if( (l >= base + 1 + iterator) && (l < base + 2*(1 + iterator)) ){
               printf("%d ", eps_mat[l]);
            }
            else{
               printf("%d ", eps_mat[l]);
            }
            // Last iteration
            if(l == block_size -1){
               printf("\n");
               // new_base = base + block_size;
               new_base = base + 3*iterator + 2;
            }
         }
      }

      else if(mem_order == 2){
         block_size = base + 5;
         for(int l = base; l < block_size; l++){
            // New line after each section
            if(l == base + 1 + 1){
               printf("\n");
            }
            else if(l == base + 1 + 1 + 1 + 1){
               printf("\n");
            }
            // 3 Sections
            if(l < 1 + 1){
               printf("%d ", eps_mat[l]);
            }
            else if( (l >= base + 1 + 1) && (l < base + 1 + 1 + 1 + 1) ){
               printf("%d ", eps_mat[l]);
            }
            else{
               printf("%d ", eps_mat[l]);
            }

            if(l == block_size -1){
               printf("\n");
               new_base = base + 5;
            }
         }
      }

      else if(mem_order == 3){
         block_size = base + 5 + 3*(iterator);
         for(int l = base; l < block_size; l++){
            // New line after each section
            if(l == base + iterator + 2){
               printf("\n");
            }
            else if(l == base + 2*iterator + 4){
               printf("\n");
            }
            // 3 Sections
            if(l < base + iterator + 2){
               printf("%d ", eps_mat[l]);
            }
            else if( (l >= base + iterator + 2) && (l < base + 2*iterator + 4) ){
               printf("%d ", eps_mat[l]);
            }
            else{
               printf("%d ", eps_mat[l]);
            }

            if(l == block_size -1){
               printf("\n");
               new_base = base + 3*iterator + 5;
            }
         }
      }
   }

}

void pt_3_sequential(dataset_t *data, uint16_t *eps_mat, int x, float eps, int counter, float iterator, int mem_order){

   float x0_s, x1_s, x2_s;
   float sum_x0x1 = 0;
   float sum_x0x2 = 0;
   float sum_x1x2 = 0;
   uint16_t d_x0_x1x2, d_x1_x2;

   for(int ftr = 0; ftr < FEATURES; ftr++){
      // 3 X
      x0_s = data[x].features[ftr];
      x1_s = data[x+1].features[ftr];
      x2_s = data[x+2].features[ftr];
      // x0,x1
      sum_x0x1 += (x0_s - x1_s)*(x0_s - x1_s);
      // x0,x2
      sum_x0x2 += (x0_s - x2_s)*(x0_s - x2_s);
      // x1,x2
      sum_x1x2 += (x1_s - x2_s)*(x1_s - x2_s);
   }
      // EPS CMP
      sum_x0x1 = (sum_x0x1 <= eps)?1:0;
      sum_x0x2 = (sum_x0x2 <= eps)?1:0;
      sum_x1x2 = (sum_x1x2 <= eps)?1:0;
         
      d_x0_x1x2 = ( ((uint16_t)sum_x0x1 << 1 ) | (uint16_t)sum_x1x2 );
      d_x1_x2 = (uint16_t)sum_x1x2;

      printf("\t3 POINT SEQUENTIAL:\n");
      printf("\td_x0_x1x2 = %d, d_x1_x2 = %d\n", d_x0_x1x2, d_x1_x2);

      // Decide how to store
      eps_mat[counter] = d_x0_x1x2;
      switch(mem_order)
      {
         case 1:
            eps_mat[counter + 1 + (int)iterator] = d_x1_x2;
         case 2:
            eps_mat[counter + 1 + 1] = d_x1_x2;
         case 3:
            eps_mat[counter + 1 + (int)iterator + 1] = d_x1_x2;
      }
}

int sequential(dataset_t *data, uint16_t *eps_mat, int x, int y, float eps, int counter, float iterator, int mem_order, int base_counter){

   float x0_s, x1_s, x2_s;
   float y_s;
   float sum_x0 = 0;
   float sum_x1 = 0;
   float sum_x2 = 0;
   uint16_t res_x0 = 0;
   uint16_t res_x1 = 0;
   uint16_t res_x2 = 0;
   int cnt_return = 0;

   for(int i = y; i < OBSERVATIONS; i++){
      for(int ftr = 0; ftr < FEATURES; ftr++){
         // 3 X
         x0_s = data[x].features[ftr];
         x1_s = data[x+1].features[ftr];
         x2_s = data[x+2].features[ftr];
         // 1 Y
         y_s = data[i].features[ftr];
         // SUB, SQR, ADD
         sum_x0 += (x0_s - y_s)*(x0_s - y_s);
         sum_x1 += (x1_s - y_s)*(x1_s - y_s);
         sum_x2 += (x2_s - y_s)*(x2_s - y_s);
      }
      // EPS CMP
      sum_x0 = (sum_x0 <= eps)?1:0;
      sum_x1 = (sum_x1 <= eps)?1:0;
      sum_x2 = (sum_x2 <= eps)?1:0;

      // Need to cast?
      res_x0 = ( (res_x0 << i) | (uint16_t)sum_x0 );
      res_x1 = ( (res_x1 << i) | (uint16_t)sum_x1 );
      res_x2 = ( (res_x2 << i) | (uint16_t)sum_x2 );

      // Reset Sums
      sum_x0 = 0;
      sum_x1 = 0;
      sum_x2 = 0;
   }
   // For the last iteration, counter was increased, so the storage is correct. 
   // For the next set of points, counter needs to point to the end of x2
   printf("\tFINAL SERIAL PORTION:\n");
   printf("\tres_x0 = %d, res_x1 = %d, res_x2 = %d\n\n", res_x0, res_x1, res_x2);

   switch(mem_order)
   {
      case 2:
         eps_mat[counter + 1] = res_x0;
         eps_mat[counter + 1 + 1 + 1] = res_x1;
         eps_mat[counter + 1 + 1 + 1 + 1] = res_x2;
         cnt_return = base_counter + 1 + 1 + 1 + 1 + 1;
      case 3:
         eps_mat[counter + 1] = res_x0;
         eps_mat[counter + 1 + (int)iterator + 1 + 1] = res_x1;
         eps_mat[counter + 1 + (int)iterator + 1 + 1 + (int)iterator + 1] = res_x2;
         cnt_return = base_counter + 1 + (int)iterator + 1 + 1 + (int)iterator + 1 + (int)iterator + 1;
   }

   return cnt_return;
}

int main(){

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
   

   {"a_aardvark", {1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1}, 1, 0},
   {"a_antelope", {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"a_bass",     {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},
   {"a_bear",     {1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1}, 1, 0},
   {"a_boar",     {1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"a_buffalo",  {1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1}, 1, 0},
   {"a_calf",     {1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1}, 1, 0},
   {"a_carp",     {0,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0}, 4, 0},
   {"a_catfish",  {0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0}, 4, 0},

};
int epsilon_matrix;

epsilon_matrix = distance(dataset);
printf("\nFINISHED DIST_CALC:%d\n", epsilon_matrix);


return 0;
}


