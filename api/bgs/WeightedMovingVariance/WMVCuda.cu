#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "WeightedMovingVarianceUtils.hpp"
#include <cstdio>

__global__ void calcWeightedVarianceMonoCuda(
                    const uint8_t *const i1, 
                    const uint8_t *const i2, 
                    const uint8_t *const i3,
                    uint8_t * o, 
                    float weight1,
                    float weight2,
                    float weight3,
                    bool enableThreshold,
                    float threshold,
                    int width,
                    int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    size_t pixIdx = row * width + col;

    float dI[] = {(float)i1[pixIdx], (float)i2[pixIdx], (float)i3[pixIdx]};
    float mean = (dI[0] * weight1) + (dI[1] * weight2) + (dI[2] * weight3);
    float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    float result = sqrtf(((value[0] * value[0]) * weight1) + ((value[1] * value[1]) * weight2) + ((value[2] * value[2]) * weight3));
    o[pixIdx] = enableThreshold ? ((uint8_t)(result > threshold ? 255.0f : 0.0f))
                                  : (uint8_t)result;
}

extern "C" void weightedVarianceMonoCuda(
    const uint8_t* const img1,
    const uint8_t* const img2,
    const uint8_t* const img3,
    uint8_t* const outImg,
    int width,
    int height,
    const WeightedMovingVarianceParams &_params)
{
    dim3 block(32, 32);
    dim3 grid(width / 32, height / 32);
    //printf("%f\n", _params.weight1);//.weight1, _params.weight2, _params.weight3);

    calcWeightedVarianceMonoCuda<<<grid, block>>>(img1, img2, img3, outImg, 
        _params.weight1, _params.weight2, _params.weight3, 
        _params.enableThreshold, _params.threshold, width, height);
    cudaThreadSynchronize();
}