#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <npp.h>
#include <nppi.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

#define CHECK_NPP(call) \
    do { \
        NppStatus err = call; \
        if (err != NPP_SUCCESS) { \
            fprintf(stderr, "NPP error in %s at line %d: %d\n", __FILE__, __LINE__, err); \
            exit(err); \
        } \
    } while (0)
