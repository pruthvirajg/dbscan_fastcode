#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

#define OBSERVATIONS 112
#define FEATURES 16

#define UNDEFINED 0
#define CLASS_1 1
#define CLASS_2 2
#define CLASS_3 3
#define CLASS_4 4
#define CLASS_5 5
#define CLASS_6 6
#define CLASS_7 7
#define NOISE 8

typedef struct dataset_t {
    char * name;
    float features[FEATURES];
    int class;
    int label;
}
dataset_t;

typedef struct neighbors_t {
    int neighbor_count;
    int neighbor[OBSERVATIONS];
}
neighbors_t;

#define EPSILON 1.70
#define MINPTS 4

void pt_3_sequential(dataset_t * data, bool * eps_mat, int x, float eps, int counter, float iterator, int mem_order);
int sequential(dataset_t * data, bool * eps_mat, int x, int seq_start, float eps, int counter, float iterator, int mem_order, int base_counter);

int distance(dataset_t * data) {

    __m256 E;
    __m256 x0, x1, x2;
    __m256 y0, y1;
    __m256 x0_y0, x0_y1, x1_y0, x1_y1;
    __m256 c0, c1, c2, c3, c4, c5;

    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    c2 = _mm256_setzero_ps();
    c3 = _mm256_setzero_ps();
    c4 = _mm256_setzero_ps();
    c5 = _mm256_setzero_ps();

    float eps = (EPSILON * EPSILON);
    int N = OBSERVATIONS;

    bool * eps_mat;

    eps_mat = (bool * ) calloc((N * N), sizeof(bool));
    if (eps_mat == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }

    //Bit-masking
    const uint8_t bit_0 = 1;
    const uint8_t bit_1 = 2;
    const uint8_t bit_2 = 4;
    const uint8_t bit_3 = 8;
    const uint8_t bit_4 = 16;
    const uint8_t bit_5 = 32;
    const uint8_t bit_6 = 64;
    const uint8_t bit_7 = 128;

    // Counts for storage 
    int counter = 0;
    int base_counter = 0;

    // 1 SIMD REG (EPSILON)
    E = _mm256_set1_ps(eps);

    // Integers for SIMD routine
    uint8_t d_x0_y0, d_x0_y1, d_x1_y0, d_x1_y1, d_x2_y0, d_x2_y1;

    // Variables for serial routine

    float iterator = 0;
    int pts_cnt;
    int mem_order;
    int next_pt;

    // Iterate over 3 X elements
    for (int x = 0; x < N; x += 3) {

        base_counter = counter;
        // Possibly define a base for all 3 points?
        // NO?  happens actually THIS WON"T HAPPEN IF N DIVISBLE BY 3
        if (x == (N - 1)) {
            break;
        }

        pts_cnt = ((N - 1) - (x + 2));
        iterator = floor(pts_cnt / 16);

        // next_pt matters for sequential
        // in Case 2, Case 3
        next_pt = x + 3;
        printf("\n");
        printf("--------------------------ITERATION x: %d--------------------\n", x);
        printf("(N-1)-(x+2) = %d\n", (N - 1) - (x + 2));
        printf("(N-1)-(x+2) %% 16 = %d\n", ((N - 1) - (x + 2)) % 16);
        printf("ITERATOR: %d\n", (int) iterator);
        

        // Decide Case for memory ordering
        // Case 1: SIMD
        // Case 2: Sequential
        // Case 3: SIMD, Sequential
        if (((pts_cnt % 16) == 0) && (pts_cnt != 0)) {
            mem_order = 1;
            pt_3_sequential(data, eps_mat, x, eps, counter, iterator, mem_order);
            next_pt = x + 3;
            // Now do SIMD
        } else if ((pts_cnt % 16) == pts_cnt) {
            mem_order = 2;
            pt_3_sequential(data, eps_mat, x, eps, counter, iterator, mem_order);
            counter = sequential(data, eps_mat, x, next_pt, eps, counter, iterator, mem_order, base_counter);
            continue;
        }
        // This means pts_cnt % 16 is equal to some number thats not 0 or pts_cnt
        else {
            mem_order = 3;
            pt_3_sequential(data, eps_mat, x, eps, counter, iterator, mem_order);
            counter = sequential(data, eps_mat, x, next_pt, eps, counter, iterator, mem_order, base_counter);
            next_pt = next_pt + (pts_cnt % 16);
            // Now do SIMD
        }
        printf("CASE %d\n", mem_order);

        // NEEDS TO BE AN IF HERE AS WELL 
        // NOW WHEN YOU ENTER THE SIMD ROUTINE
        // 1.) If only SIMD, y will go from next_pt all the way until N-1
        // 2.) If Sequential, then SIMD, it's adjusted, so it will go all the way until N-1 (WE DONT NEED TO CHECK IF y+16 < N)
        for (int y = next_pt; y < N; y += 16) {
            for (int ftr = 0; ftr < FEATURES; ftr++) {
                //------------------------------- SIMD PORTION ---------------------------------

                // 3 SIMD REG (BROADCAST POINTS)
                x0 = _mm256_set1_ps(data[x].features[ftr]);
                x1 = _mm256_set1_ps(data[x + 1].features[ftr]);
                x2 = _mm256_set1_ps(data[x + 2].features[ftr]);

                // 2 SIMD REG (DISTANCE POINTS)
                y0 = _mm256_set_ps(data[y + 7].features[ftr], data[y + 6].features[ftr], data[y + 5].features[ftr], data[y + 4].features[ftr], data[y + 3].features[ftr], data[y + 2].features[ftr], data[y + 1].features[ftr], data[y].features[ftr]);
                y1 = _mm256_set_ps(data[y + 15].features[ftr], data[y + 14].features[ftr], data[y + 13].features[ftr], data[y + 12].features[ftr], data[y + 11].features[ftr], data[y + 10].features[ftr], data[y + 9].features[ftr], data[y + 8].features[ftr]);

                // 4 SIMD REG (INTERMEDIATE SUB)
                x0_y0 = _mm256_sub_ps(x0, y0);
                x0_y1 = _mm256_sub_ps(x0, y1);
                x1_y0 = _mm256_sub_ps(x1, y0);
                x1_y1 = _mm256_sub_ps(x1, y1);
                y0 = _mm256_sub_ps(x2, y0);
                y1 = _mm256_sub_ps(x2, y1);

                // 6 SIMD REG (OUTPUT)
                c0 = _mm256_fmadd_ps(x0_y0, x0_y0, c0);
                c1 = _mm256_fmadd_ps(x0_y1, x0_y1, c1);
                c2 = _mm256_fmadd_ps(x1_y0, x1_y0, c2);
                c3 = _mm256_fmadd_ps(x1_y1, x1_y1, c3);
                c4 = _mm256_fmadd_ps(y0, y0, c4);
                c5 = _mm256_fmadd_ps(y1, y1, c5);
            }

            c0 = _mm256_cmp_ps(c0, E, _CMP_LE_OQ);
            c1 = _mm256_cmp_ps(c1, E, _CMP_LE_OQ);
            c2 = _mm256_cmp_ps(c2, E, _CMP_LE_OQ);
            c3 = _mm256_cmp_ps(c3, E, _CMP_LE_OQ);
            c4 = _mm256_cmp_ps(c4, E, _CMP_LE_OQ);
            c5 = _mm256_cmp_ps(c5, E, _CMP_LE_OQ);

            // x0 and 16 pts
            // MSB bits are the higher indices
            d_x0_y0 = _mm256_movemask_ps(c0);
            d_x0_y1 = _mm256_movemask_ps(c1);
            // x1 and 16 pts
            d_x1_y0 = _mm256_movemask_ps(c2);
            d_x1_y1 = _mm256_movemask_ps(c3);
            // x2 and 16 pts
            d_x2_y0 = _mm256_movemask_ps(c4);
            d_x2_y1 = _mm256_movemask_ps(c5);

            

            // This needs double storage, for square
            // Mem_order doensn't matter anymore
            // Whether its mem order 1 or 3, changes only the start pt i.e. y = next_pt

            for(int shift = 0; shift < 8; shift++){
                // Original triangle storage
                eps_mat[ N*x + y + shift ] = (bool)(d_x0_y0 & (1 << shift));
                eps_mat[ N*x + y + shift + 8 ] = (bool)(d_x0_y1 & (1 << shift));

                eps_mat[ N*(x+1) + y + shift ] = (bool)(d_x1_y0 & (1 << shift));
                eps_mat[ N*(x+1) + y + shift + 8 ] = (bool)(d_x1_y1 & (1 << shift));

                eps_mat[ N*(x+2) + y + shift ] = (bool)(d_x2_y0 & (1 << shift));
                eps_mat[ N*(x+2) + y + shift + 8 ] = (bool)(d_x2_y1 & (1 << shift));

                // Square storage
                eps_mat[ N*(y+shift) + x ] = (bool)(d_x0_y0 & (1 << shift));
                eps_mat[ N*(y+shift+8) + x ] = (bool)(d_x0_y1 & (1 << shift));

                eps_mat[ N*(y+shift) + x + 1 ] = (bool)(d_x1_y0 & (1 << shift));
                eps_mat[ N*(y+shift+8) + x + 1 ] = (bool)(d_x1_y1 & (1 << shift));

                eps_mat[ N*(y+shift) + x + 2 ] = (bool)(d_x2_y0 & (1 << shift));
                eps_mat[ N*(y+shift+8) + x + 2 ] = (bool)(d_x2_y1 & (1 << shift));

            }

            // Original triangle storage
            //eps_mat[] = (bool)(d_x0_y0 & bit_0);
            // eps_mat[] = (bool)(d_x0_y0 & bit_1);
            // eps_mat[] = (bool)(d_x0_y0 & bit_2);
            // eps_mat[] = (bool)(d_x0_y0 & bit_3);
            // eps_mat[] = (bool)(d_x0_y0 & bit_4);
            // eps_mat[] = (bool)(d_x0_y0 & bit_5);
            // eps_mat[] = (bool)(d_x0_y0 & bit_6);
            // eps_mat[] = (bool)(d_x0_y0 & bit_7);

            printf("\t\tSIMD PORTION WITH y = %d\n", y);
            printf("\t\t\t d_x0_y0 = %d,  d_x1_y0 = %d, d_x2_y0 = %d\n", d_x0_y0, d_x1_y0, d_x2_y0);

            //    // Decide how to store

            //    eps_mat[counter + 1] = d_x0_y0;

            //    switch(mem_order)
            //    {
            //       case 1:
            //          eps_mat[counter + 1 + (int)iterator + 1] = d_x1_y0;
            //          eps_mat[counter + 1 + (int)iterator + 1 + (int)iterator] = d_x2_y0;
            //       case 3:
            //          eps_mat[counter + 1 + (int)iterator + 1 + 1] = d_x1_y0;
            //          eps_mat[counter + 1 + (int)iterator + 1 + 1 + (int)iterator + 1] = d_x2_y0;
            //    }
            //    counter++;
            // 

            // counter tells us the offset, or delta
            counter += 16;

            // RESET C REG TO 0   
            c0 = _mm256_setzero_ps();
            c1 = _mm256_setzero_ps();
            c2 = _mm256_setzero_ps();
            c3 = _mm256_setzero_ps();
            c4 = _mm256_setzero_ps();
            c5 = _mm256_setzero_ps();
            //------------------------------- SIMD PORTION ---------------------------------
        }

        // At this point SIMD, Seq + SIMD has finished, update the counter or base if needed

    }

    printf("\n");
    for(int ind = 0; ind < N*N; ind++){
        if( (ind % N) == 0){
            printf("\n");
        }
        printf("%d ", eps_mat[ind]);
    }

    free(eps_mat);

    return 1;
}

void pt_3_sequential(dataset_t * data, bool * eps_mat, int x, float eps, int counter, float iterator, int mem_order) {

    // NOTE: MEM_ORDER AND ITERATOR AND COUNTER SHOULDNT MATTER
    float x0_s, x1_s, x2_s;
    float sum_x0x1 = 0;
    float sum_x0x2 = 0;
    float sum_x1x2 = 0;
    bool d_x0_x1, d_x0_x2, d_x1_x2;
    int N = OBSERVATIONS;

    for (int ftr = 0; ftr < FEATURES; ftr++) {
        // 3 X
        x0_s = data[x].features[ftr];
        x1_s = data[x + 1].features[ftr];
        x2_s = data[x + 2].features[ftr];
        // x0,x1
        sum_x0x1 += (x0_s - x1_s) * (x0_s - x1_s);
        // x0,x2
        sum_x0x2 += (x0_s - x2_s) * (x0_s - x2_s);
        // x1,x2
        sum_x1x2 += (x1_s - x2_s) * (x1_s - x2_s);
    }
    // EPS CMP
    sum_x0x1 = (sum_x0x1 <= eps) ? 1 : 0;
    sum_x0x2 = (sum_x0x2 <= eps) ? 1 : 0;
    sum_x1x2 = (sum_x1x2 <= eps) ? 1 : 0;

    // This is not efficient, just store directly 
    d_x0_x1 = (bool) sum_x0x1;
    d_x0_x2 = (bool) sum_x0x2;
    d_x1_x2 = (bool) sum_x1x2;

    printf("\t3 POINT SEQUENTIAL:\n");
    printf("\td_x0_x1 = %d, d_x0_x2 = %d, d_x1_x2 = %d\n", d_x0_x1, d_x0_x2, d_x1_x2);

    // Now we know,
    // Bases       Nx,          N(x+1),                N(x+2)
    // Diagonals:  Nx + x,      N(x+1) + (x+1),        N(x+2) + (x+2)
    // Write from: Nx + x + 1,  N(x+1) + (x+1) + 1,    N(x+2) + (x+2) + 1


    // MEM ORDER SHOULDNT MATTER

    // Original triangle storage
    eps_mat[ (N*x + x) + 1] = (bool) sum_x0x1;
    eps_mat[ (N*x + x) + 2] = (bool) sum_x0x2;
    eps_mat[ (N*(x+1) + x+1) + 1] = (bool) sum_x1x2;

    // Square storage
    // check if these additional spaces exist
    //a
    eps_mat[ N*(x+1) + (x+1) - 1] = (bool) sum_x0x1;
    //b
    eps_mat[ N*(x+2) + (x+2) - 2] = (bool) sum_x0x2;
    //c
    eps_mat[ N*(x+2) + (x+2) - 1] = (bool) sum_x1x2;

}

int sequential(dataset_t * data, bool * eps_mat, int x, int seq_start, float eps, int counter, float iterator, int mem_order, int base_counter) {

    float x0_s, x1_s, x2_s;
    float y_s;
    float sum_x0 = 0;
    float sum_x1 = 0;
    float sum_x2 = 0;
    uint16_t res_x0 = 0;
    uint16_t res_x1 = 0;
    uint16_t res_x2 = 0;
    int cnt_return = 0;
    int N = OBSERVATIONS;
    int seq_end = OBSERVATIONS;
    uint8_t x0_low, x0_high, x1_low, x1_high, x2_low, x2_high;
    int num_iter;
    uint16_t value_x0, value_x1, value_x2;
    int delta;
    //Bit-masking
    const uint8_t bit_0 = 1;
    const uint8_t bit_1 = 2;
    const uint8_t bit_2 = 4;
    const uint8_t bit_3 = 8;
    const uint8_t bit_4 = 16;
    const uint8_t bit_5 = 32;
    const uint8_t bit_6 = 64;
    const uint8_t bit_7 = 128;


    // Remember why you did this 
    // if its case 3, then we have to do SIMD after this, so we have to reduce N
    // if its case 2, this is the final step
    if (mem_order == 3) {
        seq_end = seq_start + ((OBSERVATIONS - 1) - (x + 2)) % 16;
        printf("\tSeq start = %d, Seq end = %d\n",seq_start, seq_end);
    }
    printf("\tOUTSIDE: CASE 2 Seq start = %d, Seq end = %d\n",seq_start, seq_end);
    num_iter = (seq_end - seq_start);
    // y = next_pt here
    // If case 2, this is all you have to calculate
    //i.e. the number of points from x+3 till N is less than 16 
    for (int i = seq_start; i < seq_end; i++) {
        for (int ftr = 0; ftr < FEATURES; ftr++) {
            // 3 X
            x0_s = data[x].features[ftr];
            x1_s = data[x + 1].features[ftr];
            x2_s = data[x + 2].features[ftr];
            // 1 Y
            y_s = data[i].features[ftr];
            // SUB, SQR, ADD
            sum_x0 += (x0_s - y_s) * (x0_s - y_s);
            sum_x1 += (x1_s - y_s) * (x1_s - y_s);
            sum_x2 += (x2_s - y_s) * (x2_s - y_s);
        }
        // EPS CMP
        //printf("BEFORE: %f %f %f\n", sum_x0, sum_x1, sum_x2);
        sum_x0 = (sum_x0 <= eps) ? 1 : 0;
        sum_x1 = (sum_x1 <= eps) ? 1 : 0;
        sum_x2 = (sum_x2 <= eps) ? 1 : 0;
        //printf("AFTER: %f %f %f\n", sum_x0, sum_x1, sum_x2);
       // printf("AFTER sum: %d %d %d\n", (uint16_t)sum_x0, (uint16_t)sum_x1, (uint16_t)sum_x2);

        // Need to cast?

        //printf("BEFORE res %d %d %d\n", res_x0, res_x1, res_x2);
        res_x0 = ((res_x0 << 1) | (uint16_t) sum_x0);
        res_x1 = ((res_x1 << 1) | (uint16_t) sum_x1);
        res_x2 = ((res_x2 << 1) | (uint16_t) sum_x2);
        //printf("AFTER res %d %d %d\n", res_x0, res_x1, res_x2);

        // Reset Sums
        sum_x0 = 0;
        sum_x1 = 0;
        sum_x2 = 0;
    }

    // For the next set of points, counter needs to point to the end of x2
    printf("\tFINAL SERIAL PORTION:\n");
    printf("\tres_x0 = %d, res_x1 = %d, res_x2 = %d\n\n", res_x0, res_x1, res_x2);


    // Inefficient, just shift out bits and multiply
    // Interpret uint16 as uint8

    // x0_low = (uint8_t) res_x0;
    // x0_high = (uint8_t)(res_x0 >> 8);
    // x1_low = (uint8_t) res_x1;
    // x1_high = (uint8_t)(res_x1 >> 8);
    // x2_low = (uint8_t) res_x2;
    // x2_high = (uint8_t)(res_x2 >> 8);


    // Now we need to know if there are less than or greater than 8 points to store
    // Nume iter will always be less than 16 
    // can define an array of the pows and just traverse 
    //printf("NUM ITER IS %d\n", num_iter);
    for(int j = 0; j < num_iter; j++){
        // +1 because you have to start from the next indice
        //printf("JERE\n");
        delta = j+1;
        // value_x0 = (res_x0 & (1 << j));
        // value_x1 = (res_x1 & (1 << j));
        // value_x2 = (res_x2 & (1 << j));
        value_x0 = (res_x0 & (1 << (num_iter-delta)));
        value_x1 = (res_x1 & (1 << (num_iter-delta)));
        value_x2 = (res_x2 & (1 << (num_iter-delta)));
        // it doens't matter if its case 2 or case 3
        // 3 pt sequential finished
        //printf("VALUE x0 = %d,VALUE x1 = %d,VALUE x2 = %d ", value_x0, value_x1,value_x2);

        // Original triangle storage
        eps_mat[ (N*x + x) + 2 + delta] = (bool) value_x0;
        eps_mat[ (N*(x+1) + x+1) + 1 + delta] = (bool) value_x1;
        eps_mat[ (N*(x+2) + x+2) + delta] = (bool) value_x2;

        // Square storage
        // re-organize this
        eps_mat[ (N*(x + 2 + delta) + x + 2 + delta) - 2 - delta] = (bool) value_x0;
        eps_mat[ (N*(x + 2 + delta) + x + 2 + delta) - 1 - delta] = (bool) value_x1;
        eps_mat[ (N*(x + 2 + delta) + x + 2 + delta) - delta] = (bool) value_x2;

    }



    cnt_return = 0;

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
printf("\n\nFINISHED DIST_CALC:%d\n", epsilon_matrix);


return 0;
}


