// Copyright 2024 EPFL
// Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// Author:  Davide Schiavone
// Date: 20/01/2025

/**
 * @file main.c
 * @brief Example of data processing (matrix multiplication) reading data from flash memory
 *
 * Simple example that read a matrix from flash memory in many step and performs
 * matrix multiplication. This is useful for applications where the
 * data size does not fit in the available SRAM memory, so some data needs to be
 * stored as "flash_only" and read trough the spi interface. This usually requires
 * filling a buffer and tiling the data processing.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "timer_sdk.h"
#include "dma_sdk.h"

/* By default, printfs are activated for FPGA and disabled for simulation. */
#define PRINTF_IN_FPGA  1
#define PRINTF_IN_SIM   0
#if TARGET_SIM && PRINTF_IN_SIM
        #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif PRINTF_IN_FPGA && !TARGET_SIM
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
    #define PRINTF(...)
#endif

#include "input_signal.h"

#define COMPUTE_LAYER_0
#define COMPUTE_LAYER_1
#define COMPUTE_LAYER_2
#define COMPUTE_LAYER_3
#define COMPUTE_LAYER_4
#define COMPUTE_LAYER_5
#define COMPUTE_LAYER_6
#define COMPUTE_LAYER_7
#define COMPUTE_LAYER_8
#define COMPUTE_LAYER_9

#define CHECK_LAYER_0
//#define CHECK_LAYER_1
//#define CHECK_LAYER_2
//#define CHECK_LAYER_3
//#define CHECK_LAYER_4
//#define CHECK_LAYER_5
//#define CHECK_LAYER_6
//#define CHECK_LAYER_7
//#define CHECK_LAYER_8
#define CHECK_LAYER_9

#include "weight0.h"
#ifdef CHECK_LAYER_0
#include "output_layer_0.h"
#endif

#include "weight1.h"
#ifdef CHECK_LAYER_1
#include "output_layer_1.h"
#endif

#include "weight2.h"
#ifdef CHECK_LAYER_2
#include "output_layer_2.h"
#endif

#include "weight3.h"
#ifdef CHECK_LAYER_3
#include "output_layer_3.h"
#endif

#include "weight4.h"
#ifdef CHECK_LAYER_4
#include "output_layer_4.h"
#endif

#include "weight5.h"
#ifdef CHECK_LAYER_5
#include "output_layer_5.h"
#endif

#include "weight6.h"
#ifdef CHECK_LAYER_6
#include "output_layer_6.h"
#endif

#include "weight7.h"
#ifdef CHECK_LAYER_7
#include "output_layer_7.h"
#endif

#include "weight8.h"
#ifdef CHECK_LAYER_8
#include "output_layer_8.h"
#endif

#include "weight9.h"
#ifdef CHECK_LAYER_9
#include "output_layer_9.h"
#endif

#define TILING_ROWS_0 32
#define TILING_ROWS_9 128
#define DO_TILING 1

#define OPENHW_GROUP_COMPILER

#ifdef OPENHW_GROUP_COMPILER
    typedef int8_t  v4qi __attribute__ ((vector_size (4)));
    #define dense8to32 dense8to32_xpulp
#else
    #define dense8to32 dense8to32_generic
#endif

int32_t __attribute__((section(".xheep_data_interleaved"))) global_matrix_buffer[WEIGHT0_ROW_] = {0};
int8_t __attribute__((section(".xheep_data_interleaved"))) output_layer_buffer0[WEIGHT0_ROW_] = {0};
int8_t __attribute__((section(".xheep_data_interleaved"))) output_layer_buffer1[WEIGHT0_ROW_] = {0};

//when we do tiling, we pretend we have a buffer in a private memory of the accelerator for fair comparison
//the size if fixed
#define TOTAL_ACC_MEMORY 32*1024

// the weight buffer must be big enough to accomodate the first matrix tiled by rows
// in anomaly detection is the biggest and is 128x640
// plus the input (INPUT_SIZE_SIGNAL)
// and the output (WEIGHT0_ROW_)
#define WEIGHT_BUFFER_L1 (TILING_ROWS_0*INPUT_SIGNAL_SIZE_)
#define INPUT_BUFFER_L1 (INPUT_SIGNAL_SIZE_)
#define OUTPUT_BUFFER_L1 (INPUT_SIGNAL_SIZE_) //neded for the last layer where autoencoder output expands

int8_t __attribute__((section(".xheep_data_interleaved_acc"))) weight_buffer_l1[WEIGHT_BUFFER_L1] = {0};
int8_t __attribute__((section(".xheep_data_interleaved_acc"))) output_layer_buffer0_l1[INPUT_BUFFER_L1] = {0};
int32_t __attribute__((section(".xheep_data_interleaved_acc"))) global_matrix_buffer_l1[OUTPUT_BUFFER_L1] = {0};
int8_t __attribute__((section(".xheep_data_interleaved_acc"))) output_layer_buffer1_l1[OUTPUT_BUFFER_L1] = {0};
//TOD: add bias buffer

#define SWAP(x, y) \
    temp = x; \
    x = y; \
    y = temp; \


#define DEF_CONCAT(x, y) x##y

#define TOTAL_TILING_SIZE(x) \
    (DEF_CONCAT(WEIGHT, x##_SIZE_) + DEF_CONCAT(INPUT_LAYER_, x##_SIZE_) + DEF_CONCAT(OUTPUT_LAYER_, x##_SIZE_) + DEF_CONCAT(OUTPUT_LAYER_, x##_SIZE_) * 4)

#define CHECK_TILING_SIZE(WEIGHT_TILE_SIZE, INPUT_TILE_SIZE_, OUTPUT_TILE_SIZE) \
    static_assert( WEIGHT_TILE_SIZE + INPUT_TILE_SIZE_ + OUTPUT_TILE_SIZE + OUTPUT_TILE_SIZE*4 <= TOTAL_ACC_MEMORY, \
                  "The tiling size is too big for the accelerator memory")


void __attribute__ ((noinline)) dense8to32_generic(int32_t* tmp_matrix32, int8_t *  A, int8_t *  B, int8_t *  C, int32_t* bias, int R1, int C2, int C1, uint8_t layer_id)
{
    for(int i = 0; i < R1; i++) {
        int32_t acc = bias[i];
        for(int k = 0; k < C1; k++) {
            acc+= A[i*C1+k] * B[k];
        }
        int8_t acc_cast = (int8_t) (acc);
        C[i] = acc_cast > 0 ? acc_cast : 0;
    }
}


#ifdef OPENHW_GROUP_COMPILER
void __attribute__ ((noinline)) dense8to32_xpulp(int32_t* tmp_matrix32, int8_t *  A, int8_t *  B, int8_t *  C, int32_t* bias, int R1, int C2, int C1, uint8_t layer_id)
{

    v4qi* av;
    v4qi* bv;

    v4qi a0;
    v4qi b0;

    for(int i = 0; i < R1; i++) {
        int32_t acc = 0;
        av = (v4qi *) &A[i*C1];
        bv = (v4qi *) &B[0];
        for(int k = 0; k < C1; k+=4) {

            a0 = *av;
            b0 = *bv;
            acc =  __builtin_riscv_cv_simd_sdotsp_b((int32_t)a0, (int32_t)b0, acc);
            av++; bv++;
        }
        C[i] = (int8_t) (acc);
    }

    av = (v4qi *) &C[0];

    for(int i = 0; i < (R1>>2); i+=2) { //>>2 /4 for SIMD, +=2 for unrolling

        a0 = av[i];
        b0 = av[i+1];
        a0 = (v4qi) __builtin_riscv_cv_simd_max_sc_b ((uint32_t)a0, 0);
        b0 = (v4qi) __builtin_riscv_cv_simd_max_sc_b ((uint32_t)b0, 0);
        av[i] = a0;
        av[i+1] = b0;
    }

}
#endif

int check_err(uint32_t output_layer_size, int8_t * act, int8_t * exp, uint8_t layer_id) {
    for(int i = 0; i < output_layer_size; i++){
        if (act[i] != exp[i]){
            printf("L%d - %d - exp : %d, act : %d\n", layer_id, i, (exp[i]), (act[i]));
            return -1;
        }
    }
    return 0;
}

int __attribute__ ((noinline)) copy_data(int8_t* src, int8_t * dst, uint32_t bytes) {
    dma_copy((uint32_t)dst, (uint32_t)src, bytes, 0, DMA_DATA_TYPE_BYTE, DMA_DATA_TYPE_BYTE, 0);
    return 0;
}

#define NUM_LAYERS 10
#define PRINT_CYCLES

int main(int argc, char *argv[]) {

    uint8_t layer_id = 0;
    int8_t* input_ptr;
    int8_t* output_ptr;
    int8_t* temp;
    uint32_t num_tiles;
    
    uint32_t cycles[NUM_LAYERS] = {0};

    dma_sdk_init();

    //check that the accelerator memory (e.g.xheep_data_interleaved_acc) is big enough to accomodate the tiling
    CHECK_TILING_SIZE(WEIGHT_BUFFER_L1, INPUT_BUFFER_L1, OUTPUT_BUFFER_L1);

    //L = 0
#ifdef COMPUTE_LAYER_0

    timer_cycles_init();
    timer_start();

    #if (TOTAL_TILING_SIZE(0) <= TOTAL_ACC_MEMORY)

        copy_data(input_signal, output_layer_buffer0_l1, INPUT_SIGNAL_SIZE_);

        input_ptr = output_layer_buffer0_l1;
        output_ptr = output_layer_buffer1_l1;

        copy_data(&weight0_w[0], weight_buffer_l1, WEIGHT0_SIZE_);
        dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight0_b, WEIGHT0_ROW_, 1, WEIGHT0_COL_,layer_id);

        cycles[layer_id] = timer_stop();

        #ifdef CHECK_LAYER_0
            if (check_err(OUTPUT_LAYER_0_SIZE_, output_ptr, output_layer_0, layer_id)!=0)
                return EXIT_FAILURE;
        #endif

        //for next layer
        SWAP(input_ptr, output_ptr);

    #else
        #ifdef DO_TILING

            num_tiles = WEIGHT0_ROW_ / TILING_ROWS_0;

            copy_data(input_signal, output_layer_buffer0_l1, INPUT_SIGNAL_SIZE_);
            input_ptr = output_layer_buffer0_l1;
            output_ptr = output_layer_buffer1_l1;

            for(int i=0; i<num_tiles;i++) {
                copy_data(&weight0_w[i*WEIGHT0_COL_*TILING_ROWS_0], weight_buffer_l1, TILING_ROWS_0*INPUT_SIGNAL_SIZE_);
                dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, &output_ptr[i*TILING_ROWS_0], &weight0_b[i*TILING_ROWS_0], TILING_ROWS_0, 1, WEIGHT0_COL_,layer_id);
            }

            cycles[layer_id] = timer_stop();

            #ifdef CHECK_LAYER_0
                if (check_err(OUTPUT_LAYER_0_SIZE_, output_ptr, output_layer_0, layer_id)!=0)
                    return EXIT_FAILURE;
            #endif

            //for next layer
            SWAP(input_ptr, output_ptr);

        #else
            #error("Tiling is required for this example")
        #endif
    #endif

#endif

#ifdef COMPUTE_LAYER_1

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 1
#if (TOTAL_TILING_SIZE(1) <= TOTAL_ACC_MEMORY)

    copy_data(&weight1_w[0], weight_buffer_l1, WEIGHT1_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight1_b, WEIGHT1_ROW_, 1, WEIGHT1_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 1")
#endif

#ifdef CHECK_LAYER_1
    if (check_err(OUTPUT_LAYER_1_SIZE_, output_ptr, output_layer_1, layer_id)!=0)
        return EXIT_FAILURE;
#endif

    //for next layer
    SWAP(input_ptr, output_ptr);

#endif

#ifdef COMPUTE_LAYER_2

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 2
#if (TOTAL_TILING_SIZE(2) <= TOTAL_ACC_MEMORY)

    copy_data(&weight2_w[0], weight_buffer_l1, WEIGHT2_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight2_b, WEIGHT2_ROW_, 1, WEIGHT2_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 2")
#endif

#ifdef CHECK_LAYER_2
    if (check_err(OUTPUT_LAYER_2_SIZE_, output_ptr, output_layer_2, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_3

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 3

#if (TOTAL_TILING_SIZE(3) <= TOTAL_ACC_MEMORY)

    copy_data(&weight3_w[0], weight_buffer_l1, WEIGHT3_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight3_b, WEIGHT3_ROW_, 1, WEIGHT3_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 3")
#endif

#ifdef CHECK_LAYER_3
    if (check_err(OUTPUT_LAYER_3_SIZE_, output_ptr, output_layer_3, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_4

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 4

#if (TOTAL_TILING_SIZE(4) <= TOTAL_ACC_MEMORY)

    copy_data(&weight4_w[0], weight_buffer_l1, WEIGHT4_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight4_b, WEIGHT4_ROW_, 1, WEIGHT4_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 4")
#endif

#ifdef CHECK_LAYER_4
    if (check_err(OUTPUT_LAYER_4_SIZE_, output_ptr, output_layer_4, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_5

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 5

#if (TOTAL_TILING_SIZE(5) <= TOTAL_ACC_MEMORY)

    copy_data(&weight5_w[0], weight_buffer_l1, WEIGHT5_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight5_b, WEIGHT5_ROW_, 1, WEIGHT5_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 5")
#endif

#ifdef CHECK_LAYER_5
    if (check_err(OUTPUT_LAYER_5_SIZE_, output_ptr, output_layer_5, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_6

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 6

#if (TOTAL_TILING_SIZE(6) <= TOTAL_ACC_MEMORY)

    copy_data(&weight6_w[0], weight_buffer_l1, WEIGHT6_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight6_b, WEIGHT6_ROW_, 1, WEIGHT6_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 6")
#endif

#ifdef CHECK_LAYER_6
    if (check_err(OUTPUT_LAYER_6_SIZE_, output_ptr, output_layer_6, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_7

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 7

#if (TOTAL_TILING_SIZE(7) <= TOTAL_ACC_MEMORY)

    copy_data(&weight7_w[0], weight_buffer_l1, WEIGHT7_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight7_b, WEIGHT7_ROW_, 1, WEIGHT7_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 7")
#endif

#ifdef CHECK_LAYER_7
    if (check_err(OUTPUT_LAYER_7_SIZE_, output_ptr, output_layer_7, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_8

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 8

#if (TOTAL_TILING_SIZE(8) <= TOTAL_ACC_MEMORY)

    copy_data(&weight8_w[0], weight_buffer_l1, WEIGHT8_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight8_b, WEIGHT8_ROW_, 1, WEIGHT8_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#else
    #error("Tiling is not implemented for layer 8")
#endif

#ifdef CHECK_LAYER_8
    if (check_err(OUTPUT_LAYER_8_SIZE_, output_ptr, output_layer_8, layer_id)!=0)
        return EXIT_FAILURE;
#endif
    //for next layer
    SWAP(input_ptr, output_ptr);
#endif

#ifdef COMPUTE_LAYER_9

    timer_cycles_init();
    timer_start();

    layer_id++; //L = 9

#if (TOTAL_TILING_SIZE(9) <= TOTAL_ACC_MEMORY)

    copy_data(&weight9_w[0], weight_buffer_l1, WEIGHT9_SIZE_);
    dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, output_ptr, weight9_b, WEIGHT9_ROW_, 1, WEIGHT9_COL_,layer_id);

    cycles[layer_id] = timer_stop();

#ifdef CHECK_LAYER_9
    if (check_err(OUTPUT_LAYER_9_SIZE_, output_ptr, output_layer_9, layer_id)!=0)
        return EXIT_FAILURE;
#endif

#else

    #ifdef DO_TILING

        num_tiles = WEIGHT9_ROW_ / TILING_ROWS_9;

        static_assert( TILING_ROWS_9*INPUT_LAYER_9_SIZE_ <= WEIGHT_BUFFER_L1, \
        "The tiling size for layer 9 is too big for the weight buffer");

        for(int i=0; i<num_tiles;i++) {
            copy_data(&weight9_w[i*WEIGHT9_COL_*TILING_ROWS_9], weight_buffer_l1, TILING_ROWS_9*INPUT_LAYER_9_SIZE_);
            dense8to32(global_matrix_buffer_l1, weight_buffer_l1, input_ptr, &output_ptr[i*TILING_ROWS_9], &weight9_b[i*TILING_ROWS_9], TILING_ROWS_9, 1, WEIGHT9_COL_,layer_id);
        }

        cycles[layer_id] = timer_stop();

        #ifdef CHECK_LAYER_9
            if (check_err(OUTPUT_LAYER_9_SIZE_, output_ptr, output_layer_9, layer_id)!=0)
                return EXIT_FAILURE;
        #endif

    #else
        #error("Tiling is required for this example")
    #endif

#endif
#endif

#ifdef PRINT_CYCLES
    for(int i = 0; i < NUM_LAYERS; i++) {
        printf("L%d cc: %d\n", i, cycles[i]);
    }
#endif

    return EXIT_SUCCESS;

}
