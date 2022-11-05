#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

#include "../include/config.h"
#include "../include/dbscan.h"
#include "../include/acc_distance.h"

void acc_distance_simd(void) {

    // For epsilon compare
    __m256 E;
    // 3 X points
    __m256 x0, x1, x2;
    // 2 Y points
    __m256 y0, y1;
    // 6 distances (4 shown due to register re-use)
    __m256 x0_y0, x0_y1, x1_y0, x1_y1;
    // Comparison results
    __m256 c0, c1, c2, c3, c4, c5;

    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    c2 = _mm256_setzero_ps();
    c3 = _mm256_setzero_ps();
    c4 = _mm256_setzero_ps();
    c5 = _mm256_setzero_ps();

    int N = TOTAL_OBSERVATIONS;

    //Bit-masking, consider using this as an array
    // const uint8_t bit_0 = 1;
    // const uint8_t bit_1 = 2;
    // const uint8_t bit_2 = 4;
    // const uint8_t bit_3 = 8;
    // const uint8_t bit_4 = 16;
    // const uint8_t bit_5 = 32;
    // const uint8_t bit_6 = 64;
    // const uint8_t bit_7 = 128;


    // 1 SIMD REG (EPSILON)
    E = _mm256_set1_ps(EPSILON_SQUARE);

    // Integers for SIMD routine (output of movemask)
    uint8_t d_x0_y0, d_x0_y1, d_x1_y0, d_x1_y1, d_x2_y0, d_x2_y1;

    // Variables for serial routine
    // iterator not needed, just for comprehension
    float iterator = 0;
    int pts_cnt;
    int mem_order = 0;
    int next_pt;

    // Iterate over 3 X elements
    // Make this N-1
    for (int x = 0; x < N; x += 3) {

        // Possibly define a base for all 3 points?
        // NO?  happens actually THIS WON"T HAPPEN IF N DIVISBLE BY 3
        if (x == (N - 1)) {
            break;
        }
        // pts_cnt tells us how many points of common calculation are there after the 3 pt seq. op.
        pts_cnt = ((N - 1) - (x + 2));
        // number of SIMD routines possible
        iterator = floor(pts_cnt / 16);

        // next_pt matters for case 2,3 which indicates the start of the serial portion
        next_pt = x + 3;

        // Just some print out stuff
        #ifdef DEBUG_ACC_DIST
        printf("\n");
        printf("--------------------------ITERATION x: %d--------------------\n", x);
        printf("(N-1)-(x+2) = %d\n", (N - 1) - (x + 2));
        printf("(N-1)-(x+2) %% 16 = %d\n", ((N - 1) - (x + 2)) % 16);
        printf("ITERATOR: %d\n", (int) iterator);
        #endif
        

        // Decide Case for memory ordering
        // Case 1: SIMD
        // Case 2: Sequential
        // Case 3: SIMD, Sequential
        if (((pts_cnt % 16) == 0) && (pts_cnt != 0)) {
            mem_order = 1;
            pt_3_sequential(x, EPSILON_SQUARE);
            // SIMD should start right after the 3pt seq. mark
            next_pt = x + 3;
            // Now do SIMD
        } else if ((pts_cnt % 16) == pts_cnt) {
            mem_order = 2;
            pt_3_sequential(x, EPSILON_SQUARE);
            sequential(x, next_pt, EPSILON_SQUARE, mem_order);
            // Next loop iteration if purely sequential
            continue;
        }
        // This means pts_cnt % 16 is equal to some number thats not 0 or pts_cnt
        else {
            mem_order = 3;
            pt_3_sequential(x, EPSILON_SQUARE);
            sequential(x, next_pt, EPSILON_SQUARE, mem_order);
            // SIMD should start after 3 pt seq + serial portion
            next_pt = next_pt + (pts_cnt % 16);
            // Now do SIMD
        }
        #ifdef DEBUG_ACC_DIST
        printf("CASE %d\n", mem_order);
        #endif
        // Generic SIMD routine for both case 1 and 3
        // They're uniquely identified by just next_pt
        for (int y = next_pt; y < N; y += 16) {
            for (int ftr = 0; ftr < FEATURES; ftr++) {
                //------------------------------- SIMD PORTION ---------------------------------

                // 3 SIMD REG (BROADCAST POINTS)
                x0 = _mm256_set1_ps(dataset[x].features[ftr]);
                x1 = _mm256_set1_ps(dataset[x + 1].features[ftr]);
                x2 = _mm256_set1_ps(dataset[x + 2].features[ftr]);

                // 2 SIMD REG (DISTANCE POINTS)
                y0 = _mm256_set_ps(dataset[y + 7].features[ftr], dataset[y + 6].features[ftr], dataset[y + 5].features[ftr], dataset[y + 4].features[ftr], dataset[y + 3].features[ftr], dataset[y + 2].features[ftr], dataset[y + 1].features[ftr], dataset[y].features[ftr]);
                y1 = _mm256_set_ps(dataset[y + 15].features[ftr], dataset[y + 14].features[ftr], dataset[y + 13].features[ftr], dataset[y + 12].features[ftr], dataset[y + 11].features[ftr], dataset[y + 10].features[ftr], dataset[y + 9].features[ftr], dataset[y + 8].features[ftr]);

                // 4 SIMD REG (INTERMEDIATE SUB)
                x0_y0 = _mm256_sub_ps(x0, y0);
                x0_y1 = _mm256_sub_ps(x0, y1);
                x1_y0 = _mm256_sub_ps(x1, y0);
                x1_y1 = _mm256_sub_ps(x1, y1);
                y0 = _mm256_sub_ps(x2, y0);
                y1 = _mm256_sub_ps(x2, y1);

                // 6 SIMD REG (FMA) (OUTPUT)
                c0 = _mm256_fmadd_ps(x0_y0, x0_y0, c0);
                c1 = _mm256_fmadd_ps(x0_y1, x0_y1, c1);
                c2 = _mm256_fmadd_ps(x1_y0, x1_y0, c2);
                c3 = _mm256_fmadd_ps(x1_y1, x1_y1, c3);
                c4 = _mm256_fmadd_ps(y0, y0, c4);
                c5 = _mm256_fmadd_ps(y1, y1, c5);
            }

            // COMPARE
            c0 = _mm256_cmp_ps(c0, E, _CMP_LE_OQ);
            c1 = _mm256_cmp_ps(c1, E, _CMP_LE_OQ);
            c2 = _mm256_cmp_ps(c2, E, _CMP_LE_OQ);
            c3 = _mm256_cmp_ps(c3, E, _CMP_LE_OQ);
            c4 = _mm256_cmp_ps(c4, E, _CMP_LE_OQ);
            c5 = _mm256_cmp_ps(c5, E, _CMP_LE_OQ);

            // MSB bits are the higher indices
            // x0 and 16 pts
            d_x0_y0 = _mm256_movemask_ps(c0);
            d_x0_y1 = _mm256_movemask_ps(c1);
            // x1 and 16 pts
            d_x1_y0 = _mm256_movemask_ps(c2);
            d_x1_y1 = _mm256_movemask_ps(c3);
            // x2 and 16 pts
            d_x2_y0 = _mm256_movemask_ps(c4);
            d_x2_y1 = _mm256_movemask_ps(c5);

            // Case 1 and 3 behave the same over here
            for(int shift = 0; shift < 8; shift++){
                // Original triangle storage
                epsilon_matrix[ N*x + y + shift ] = (bool)(d_x0_y0 & (1 << shift));
                epsilon_matrix[ N*x + y + shift + 8 ] = (bool)(d_x0_y1 & (1 << shift));

                epsilon_matrix[ N*(x+1) + y + shift ] = (bool)(d_x1_y0 & (1 << shift));
                epsilon_matrix[ N*(x+1) + y + shift + 8 ] = (bool)(d_x1_y1 & (1 << shift));

                epsilon_matrix[ N*(x+2) + y + shift ] = (bool)(d_x2_y0 & (1 << shift));
                epsilon_matrix[ N*(x+2) + y + shift + 8 ] = (bool)(d_x2_y1 & (1 << shift));

                // Square storage
                epsilon_matrix[ N*(y+shift) + x ] = (bool)(d_x0_y0 & (1 << shift));
                epsilon_matrix[ N*(y+shift+8) + x ] = (bool)(d_x0_y1 & (1 << shift));

                epsilon_matrix[ N*(y+shift) + x + 1 ] = (bool)(d_x1_y0 & (1 << shift));
                epsilon_matrix[ N*(y+shift+8) + x + 1 ] = (bool)(d_x1_y1 & (1 << shift));

                epsilon_matrix[ N*(y+shift) + x + 2 ] = (bool)(d_x2_y0 & (1 << shift));
                epsilon_matrix[ N*(y+shift+8) + x + 2 ] = (bool)(d_x2_y1 & (1 << shift));

            }
            #ifdef DEBUG_ACC_DIST
            printf("\t\tSIMD PORTION WITH y = %d\n", y);
            printf("\t\t\t d_x0_y0 = %d,  d_x1_y0 = %d, d_x2_y0 = %d\n", d_x0_y0, d_x1_y0, d_x2_y0);
            #endif
            // RESET C REG TO 0   
            c0 = _mm256_setzero_ps();
            c1 = _mm256_setzero_ps();
            c2 = _mm256_setzero_ps();
            c3 = _mm256_setzero_ps();
            c4 = _mm256_setzero_ps();
            c5 = _mm256_setzero_ps();
            //------------------------------- SIMD PORTION ---------------------------------
        }
    }
}

void pt_3_sequential(int x, float eps) {

    float x0_s, x1_s, x2_s;
    float sum_x0x1 = 0;
    float sum_x0x2 = 0;
    float sum_x1x2 = 0;
    int N = OBSERVATIONS;

    for (int ftr = 0; ftr < FEATURES; ftr++) {
        // 3 X
        x0_s = dataset[x].features[ftr];
        x1_s = dataset[x + 1].features[ftr];
        x2_s = dataset[x + 2].features[ftr];
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
    #ifdef DEBUG_ACC_DIST
    printf("\t3 POINT SEQUENTIAL:\n");
    printf("\tsum_x0x1 = %d, sum_x0x2 = %d, sum_x1x2 = %d\n", (bool)sum_x0x1, (bool)sum_x0x2, (bool)sum_x1x2);
    #endif
    // Now we know,
    // Bases       Nx,          N(x+1),                N(x+2)
    // Diagonals:  Nx + x,      N(x+1) + (x+1),        N(x+2) + (x+2)
    // Write from: Nx + x + 1,  N(x+1) + (x+1) + 1,    N(x+2) + (x+2) + 1

    // Original triangle storage
    epsilon_matrix[ (N*x + x) + 1] = (bool) sum_x0x1;
    epsilon_matrix[ (N*x + x) + 2] = (bool) sum_x0x2;
    epsilon_matrix[ (N*(x+1) + x+1) + 1] = (bool) sum_x1x2;

    // Square storage
    // check if these additional spaces exist, could have a corner case when N != 112
    epsilon_matrix[ N*(x+1) + (x+1) - 1] = (bool) sum_x0x1;
    epsilon_matrix[ N*(x+2) + (x+2) - 2] = (bool) sum_x0x2;
    epsilon_matrix[ N*(x+2) + (x+2) - 1] = (bool) sum_x1x2;

}

void sequential(int x, int seq_start, float eps, int mem_order) {

    float x0_s, x1_s, x2_s;
    float y_s;
    float sum_x0 = 0;
    float sum_x1 = 0;
    float sum_x2 = 0;
    uint16_t res_x0 = 0;
    uint16_t res_x1 = 0;
    uint16_t res_x2 = 0;
    int N = OBSERVATIONS;
    int seq_end = OBSERVATIONS;
    int num_iter;
    uint16_t value_x0, value_x1, value_x2;
    int delta;
    //Bit-masking, consider using array
    // const uint8_t bit_0 = 1;
    // const uint8_t bit_1 = 2;
    // const uint8_t bit_2 = 4;
    // const uint8_t bit_3 = 8;
    // const uint8_t bit_4 = 16;
    // const uint8_t bit_5 = 32;
    // const uint8_t bit_6 = 64;
    // const uint8_t bit_7 = 128;

    // If its case 3, then the end point is the point after which everything can be done in pure SIMD, so update the end
    // If its case 2, this is the final step, so let the end be N itself
    if (mem_order == 3) {
        seq_end = seq_start + ((OBSERVATIONS - 1) - (x + 2)) % 16;
        #ifdef DEBUG_ACC_DIST
        printf("\tSeq start = %d, Seq end = %d\n",seq_start, seq_end);
        #endif
    }
    #ifdef DEBUG_ACC_DIST
    printf("\tOUTSIDE: CASE 2 Seq start = %d, Seq end = %d\n",seq_start, seq_end);
    #endif
    num_iter = (seq_end - seq_start);

    for (int i = seq_start; i < seq_end; i++) {
        for (int ftr = 0; ftr < FEATURES; ftr++) {
            // 3 X
            x0_s = dataset[x].features[ftr];
            x1_s = dataset[x + 1].features[ftr];
            x2_s = dataset[x + 2].features[ftr];
            // 1 Y
            y_s = dataset[i].features[ftr];
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
        // printf("AFTER sum: %d %d %d\n", (uint16_t)sum_x0, (uint16_t)sum_x1, (uint16_t)sum_x2);

        // HAVE TO CAST
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
    #ifdef DEBUG_ACC_DIST
    printf("\tFINAL SERIAL PORTION:\n");
    printf("\tres_x0 = %d, res_x1 = %d, res_x2 = %d\n\n", res_x0, res_x1, res_x2);
    #endif


    // Now we need to know if there are less than or greater than 8 points to store
    // Nume iter will always be less than 16 
    //printf("NUM ITER IS %d\n", num_iter);
    for(int j = 0; j < num_iter; j++){
        // +1 because you have to start from the next indice
        delta = j+1;
        value_x0 = (res_x0 & (1 << (num_iter-delta)));
        value_x1 = (res_x1 & (1 << (num_iter-delta)));
        value_x2 = (res_x2 & (1 << (num_iter-delta)));
        // 3 pt sequential finished
        //printf("VALUE x0 = %d,VALUE x1 = %d,VALUE x2 = %d ", value_x0, value_x1,value_x2);

        // Original triangle storage
        epsilon_matrix[ (N*x + x) + 2 + delta] = (bool) value_x0;
        epsilon_matrix[ (N*(x+1) + x+1) + 1 + delta] = (bool) value_x1;
        epsilon_matrix[ (N*(x+2) + x+2) + delta] = (bool) value_x2;

        // Square storage
        // re-organize this if needed
        epsilon_matrix[ (N*(x + 2 + delta) + x + 2 + delta) - 2 - delta] = (bool) value_x0;
        epsilon_matrix[ (N*(x + 2 + delta) + x + 2 + delta) - 1 - delta] = (bool) value_x1;
        epsilon_matrix[ (N*(x + 2 + delta) + x + 2 + delta) - delta] = (bool) value_x2;
    }
}

