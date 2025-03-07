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

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


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
    // Set to 0
    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    c2 = _mm256_setzero_ps();
    c3 = _mm256_setzero_ps();
    c4 = _mm256_setzero_ps();
    c5 = _mm256_setzero_ps();

    int N = TOTAL_OBSERVATIONS;

    // 1 SIMD REG (EPSILON)
    E = _mm256_set1_ps(EPSILON_SQUARE);

    // Integers for SIMD routine (output of movemask)
    uint8_t d_x0_y0, d_x0_y1, d_x1_y0, d_x1_y1, d_x2_y0, d_x2_y1;

    // Variables for sequential routine
    int pts_cnt;
    int next_pt;

    // Iterate over 3 X elements
    for (int x = 0; x < N-1; x += 3) {

        // pts_cnt tells us how many points of common calculation are there after the pt_3_seq
        pts_cnt = ((N - 1) - (x + 2));
 
        // next_pt matters for case 2,3 which indicates the start of the serial portion
        next_pt = x + 3;

        // Just some print out stuff
        #ifdef DEBUG_ACC_DIST
        printf("\n--------------------------ITERATION x: %d--------------------------\n", x);
        printf("(N-1)-(x+2) = %d\n", (N - 1) - (x + 2));
        printf("(N-1)-(x+2) %% 16 = %d\n", ((N - 1) - (x + 2)) % 16);
        #endif
        
        // Decide Case for memory ordering
        // Case 1: SIMD
        // Case 2: Sequential
        // Case 3: SIMD, Sequential
        pt_3_sequential(x);
        if (((pts_cnt % 16) == 0) && (pts_cnt != 0)) {
            // SIMD should start right after the pt_3_seq
            next_pt = x + 3;
            // Now do SIMD
        } else if ((pts_cnt % 16) == pts_cnt) {
            sequential(x, next_pt, pts_cnt);
            // Next loop iteration if purely sequential
            continue;
        }
        // This means pts_cnt % 16 is equal to some number thats not 0 or pts_cnt
        else {
            sequential(x, next_pt, pts_cnt);
            // SIMD should start after pt_3_seq + sequential portion
            next_pt = next_pt + (pts_cnt % 16);
            // Now do SIMD
        }

        // Generic SIMD routine for both case 1 and 3
        // They're uniquely identified by just next_pt
        #ifdef BENCHMARK_SIMD
        simd_dst_call_count += ((N - next_pt)/ 16) * FEATURES;

        // simd_dst_st = rdtsc();
        #endif
        DTYPE *x0_farr, *x1_farr, *x2_farr;
        DTYPE *y0_farr, *y1_farr, *y2_farr, *y3_farr, *y4_farr, *y5_farr, *y6_farr, *y7_farr;
        DTYPE *y8_farr, *y9_farr, *y10_farr, *y11_farr, *y12_farr, *y13_farr, *y14_farr, *y15_farr;

        for (int y = next_pt; y < N; y += 16) {
            #ifdef BENCHMARK_SIMD
            simd_dst_st = rdtsc();
            #endif

            x0_farr = dataset[x].features;
            x1_farr = dataset[x + 1].features;
            x2_farr = dataset[x + 2].features;

            y0_farr = dataset[y].features;
            y1_farr = dataset[y + 1].features;
            y2_farr = dataset[y + 2].features;
            y3_farr = dataset[y + 3].features;
            y4_farr = dataset[y + 4].features;
            y5_farr = dataset[y + 5].features;
            y6_farr = dataset[y + 6].features;
            y7_farr = dataset[y + 7].features;

            y8_farr = dataset[y + 8].features;
            y9_farr = dataset[y + 9].features;
            y10_farr = dataset[y + 10].features;
            y11_farr = dataset[y + 11].features;
            y12_farr = dataset[y + 12].features;
            y13_farr = dataset[y + 13].features;
            y14_farr = dataset[y + 14].features;
            y15_farr = dataset[y + 15].features;

            for (int ftr = 0; ftr < FEATURES; ftr++) {
                //------------------------------- SIMD PORTION ---------------------------------

                // 3 SIMD REG (BROADCAST POINTS)
                x0 = _mm256_set1_ps(x0_farr[ftr]);
                x1 = _mm256_set1_ps(x1_farr[ftr]);
                x2 = _mm256_set1_ps(x2_farr[ftr]);

                // 2 SIMD REG (DISTANCE POINTS)
                // y0 = _mm256_set_ps(dataset[y + 7].features[ftr], dataset[y + 6].features[ftr], dataset[y + 5].features[ftr], dataset[y + 4].features[ftr], dataset[y + 3].features[ftr], dataset[y + 2].features[ftr], dataset[y + 1].features[ftr], dataset[y].features[ftr]);
                // y1 = _mm256_set_ps(dataset[y + 15].features[ftr], dataset[y + 14].features[ftr], dataset[y + 13].features[ftr], dataset[y + 12].features[ftr], dataset[y + 11].features[ftr], dataset[y + 10].features[ftr], dataset[y + 9].features[ftr], dataset[y + 8].features[ftr]);

                y0 = _mm256_set_ps(y7_farr[ftr], y6_farr[ftr], y5_farr[ftr], y4_farr[ftr], y3_farr[ftr], y2_farr[ftr], y1_farr[ftr], y0_farr[ftr]);
                y1 = _mm256_set_ps(y15_farr[ftr], y14_farr[ftr], y13_farr[ftr], y12_farr[ftr], y11_farr[ftr], y10_farr[ftr], y9_farr[ftr], y8_farr[ftr]);
                
                // // 4 SIMD REG (INTERMEDIATE SUB)
                // x0_y0 = _mm256_sub_ps(x0, y0);
                // x0_y1 = _mm256_sub_ps(x0, y1);
                // x1_y0 = _mm256_sub_ps(x1, y0);
                // x1_y1 = _mm256_sub_ps(x1, y1);
                // y0 = _mm256_sub_ps(x2, y0);
                // y1 = _mm256_sub_ps(x2, y1);

                // // 6 SIMD REG (FMA) (OUTPUT)
                // c0 = _mm256_fmadd_ps(x0_y0, x0_y0, c0);
                // c1 = _mm256_fmadd_ps(x0_y1, x0_y1, c1);
                // c2 = _mm256_fmadd_ps(x1_y0, x1_y0, c2);
                // c3 = _mm256_fmadd_ps(x1_y1, x1_y1, c3);
                // c4 = _mm256_fmadd_ps(y0, y0, c4);
                // c5 = _mm256_fmadd_ps(y1, y1, c5);

                // 4 SIMD REG (INTERMEDIATE SUB)
                x0_y0 = _mm256_sub_ps(x0, y0);
                x0_y1 = _mm256_sub_ps(x0, y1);
                x1_y0 = _mm256_sub_ps(x1, y0);

                c0 = _mm256_fmadd_ps(x0_y0, x0_y0, c0);
                c1 = _mm256_fmadd_ps(x0_y1, x0_y1, c1);
                c2 = _mm256_fmadd_ps(x1_y0, x1_y0, c2);

                x1_y1 = _mm256_sub_ps(x1, y1);
                y0 = _mm256_sub_ps(x2, y0);
                y1 = _mm256_sub_ps(x2, y1);

                // 6 SIMD REG (FMA) (OUTPUT)
                c3 = _mm256_fmadd_ps(x1_y1, x1_y1, c3);
                c4 = _mm256_fmadd_ps(y0, y0, c4);
                c5 = _mm256_fmadd_ps(y1, y1, c5);
            }

            #ifdef BENCHMARK_SIMD
            simd_dst_et = rdtsc();
            simd_dst_cycles += (simd_dst_et - simd_dst_st);
            #endif
            
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
            // TODO: Explore conditional storage: don't store if 0 (?)
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

        // #ifdef BENCHMARK_SIMD
        // simd_dst_et = rdtsc();
        // simd_dst_cycles += (simd_dst_et - simd_dst_st);
        // #endif
    }
}

void pt_3_sequential(int x) {

    float x0_s, x1_s, x2_s;
    float sum_x0x1 = 0;
    float sum_x0x2 = 0;
    float sum_x1x2 = 0;
    int N = TOTAL_OBSERVATIONS;

    for (int ftr = 0; ftr < FEATURES; ftr++) {
        // 3 X
        x0_s = dataset[x].features[ftr];
        x1_s = dataset[x + 1].features[ftr];
        x2_s = dataset[x + 2].features[ftr];
        // x0,x1
        sum_x0x1 += pow( (x0_s - x1_s), 2);
        // x0,x2
        sum_x0x2 += pow( (x0_s - x2_s), 2);
        // x1,x2
        sum_x1x2 += pow( (x1_s - x2_s), 2);
    }
    // EPS CMP
    sum_x0x1 = (sum_x0x1 <= EPSILON_SQUARE) ? 1 : 0;
    sum_x0x2 = (sum_x0x2 <= EPSILON_SQUARE) ? 1 : 0;
    sum_x1x2 = (sum_x1x2 <= EPSILON_SQUARE) ? 1 : 0;

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
    // TODO: check if these additional spaces exist, could have a corner case when N != 112
    epsilon_matrix[ N*(x+1) + (x+1) - 1] = (bool) sum_x0x1;
    epsilon_matrix[ N*(x+2) + (x+2) - 2] = (bool) sum_x0x2;
    epsilon_matrix[ N*(x+2) + (x+2) - 1] = (bool) sum_x1x2;
}

void sequential(int x, int seq_start, int pts_cnt) {

    float x0_s, x1_s, x2_s;
    float y_s;
    float sum_x0 = 0;
    float sum_x1 = 0;
    float sum_x2 = 0;
    int N = TOTAL_OBSERVATIONS;
    int seq_end;
    int delta = 1;

    seq_end = seq_start + (pts_cnt % 16);

    #ifdef DEBUG_ACC_DIST
    printf("\tSeq start = %d, Seq end = %d\n",seq_start, seq_end);
    #endif

    for (int i = seq_start; i < seq_end; i++) {
        for (int ftr = 0; ftr < FEATURES; ftr++) {
            // 3 X
            x0_s = dataset[x].features[ftr];
            x1_s = dataset[x + 1].features[ftr];
            x2_s = dataset[x + 2].features[ftr];
            // 1 Y
            y_s = dataset[i].features[ftr];
            // SUB, SQR, ADD
            sum_x0 += pow((x0_s - y_s), 2);
            sum_x1 += pow((x1_s - y_s), 2);
            sum_x2 += pow((x2_s - y_s), 2);
        }
        // EPS CMP
        sum_x0 = (sum_x0 <= EPSILON_SQUARE) ? 1 : 0;
        sum_x1 = (sum_x1 <= EPSILON_SQUARE) ? 1 : 0;
        sum_x2 = (sum_x2 <= EPSILON_SQUARE) ? 1 : 0;

        // Original triangle storage
        epsilon_matrix[ (N*x + x) + 2 + delta] = (bool) sum_x0;
        epsilon_matrix[ (N*(x+1) + x+1) + 1 + delta] = (bool) sum_x1;
        epsilon_matrix[ (N*(x+2) + x+2) + delta] = (bool) sum_x2;

        // Square storage
        epsilon_matrix[ (N*(x + 2 + delta) + x + 2 + delta) - 2 - delta] = (bool) sum_x0;
        epsilon_matrix[ (N*(x + 2 + delta) + x + 2 + delta) - 1 - delta] = (bool) sum_x1;
        epsilon_matrix[ (N*(x + 2 + delta) + x + 2 + delta) - delta] = (bool) sum_x2;
        delta++;

        // Reset Sums
        sum_x0 = 0;
        sum_x1 = 0;
        sum_x2 = 0;
    }
}
