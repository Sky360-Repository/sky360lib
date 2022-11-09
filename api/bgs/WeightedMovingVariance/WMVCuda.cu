#include <stdint.h>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include "core.hpp"
#include "WeightedMovingVarianceUtils.hpp"

__global__ void calcWeightedVarianceMonoCuda(
                    const uint8_t *const i1, 
                    const uint8_t *const i2, 
                    const uint8_t *const i3,
                    uint8_t * o, 
                    float weight1,
                    float weight2,
                    float weight3)
{
    size_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

    float dI[] = {(float)i1[pixIdx], (float)i2[pixIdx], (float)i3[pixIdx]};
    float mean = (dI[0] * weight1) + (dI[1] * weight2) + (dI[2] * weight3);
    float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    o[pixIdx] = sqrtf((value[0] * value[0]) * weight1) + ((value[1] * value[1]) * weight2) + ((value[2] * value[2]) * weight3);
}

__global__ void calcWeightedVarianceMonoThresholdCuda(
                    const uint8_t *const i1, 
                    const uint8_t *const i2, 
                    const uint8_t *const i3,
                    uint8_t * o, 
                    float w1,
                    float w2,
                    float w3,
                    float threshold)
{
    const size_t pixIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 100;
    const float w[]{w1, w2, w3};

    for (int i{0}; i < 100; ++i)
    {
        float dI[] = {(float)i1[pixIdx + i], (float)i2[pixIdx + i], (float)i3[pixIdx + i]};
        float mean = (dI[0] * w[0]) + (dI[1] * w[1]) + (dI[2] * w[2]);
        float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
        float result = ((value[0] * value[0]) * w[0]) + ((value[1] * value[1]) * w[1]) + ((value[2] * value[2]) * w[2]);
        o[pixIdx + i] = result > threshold ? MAX_UC : ZERO_UC;
    }
}

extern "C" void weightedVarianceMonoCuda(
    const uint8_t* const img1,
    const uint8_t* const img2,
    const uint8_t* const img3,
    uint8_t* const outImg,
    const size_t numPixels,
    const WeightedMovingVarianceParams &_params)
{
    const size_t numThreads{1024};
    const size_t numBlocks{numPixels / numThreads / 100};
    //printf("%f\n", _params.weight1);//.weight1, _params.weight2, _params.weight3);

    if (_params.enableThreshold)
        calcWeightedVarianceMonoThresholdCuda<<<numBlocks, numThreads>>>(img1, img2, img3, outImg, 
            _params.weight[0], _params.weight[1], _params.weight[2], _params.thresholdSquared);
    else
        calcWeightedVarianceMonoCuda<<<numBlocks, numThreads>>>(img1, img2, img3, outImg, 
            _params.weight[0], _params.weight[1], _params.weight[2]);

    cudaThreadSynchronize();
}

__global__ void calcWeightedVarianceColorThreshold(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                      uint8_t *const o, float weight1, float weight2, float weight3, float threshold)
{
    size_t pixIdx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t pixIdx3{pixIdx * 3};

    const float dI1[] = {(float)i1[pixIdx3], (float)i1[pixIdx3] + 1, (float)i1[pixIdx3 + 2]};
    const float dI2[] = {(float)i2[pixIdx3], (float)i2[pixIdx3] + 1, (float)i2[pixIdx3 + 2]};
    const float dI3[] = {(float)i3[pixIdx3], (float)i3[pixIdx3] + 1, (float)i3[pixIdx3 + 2]};
    const float meanR{(dI1[0] * weight1) + (dI2[0] * weight2) + (dI3[0] * weight3)};
    const float meanG{(dI1[1] * weight1) + (dI2[1] * weight2) + (dI3[1] * weight3)};
    const float meanB{(dI1[2] * weight1) + (dI2[2] * weight2) + (dI3[2] * weight3)};
    const float valueR[] = {dI1[0] - meanR, dI2[0] - meanR, dI2[0] - meanR};
    const float valueG[] = {dI1[1] - meanG, dI2[1] - meanG, dI2[1] - meanG};
    const float valueB[] = {dI1[2] - meanB, dI2[2] - meanB, dI2[2] - meanB};
    const float r2{((valueR[0] * valueR[0]) * weight1) + ((valueR[1] * valueR[1]) * weight2) + ((valueR[2] * valueR[2]) * weight3)};
    const float g2{((valueG[0] * valueG[0]) * weight1) + ((valueG[1] * valueG[1]) * weight2) + ((valueG[2] * valueG[2]) * weight3)};
    const float b2{((valueB[0] * valueB[0]) * weight1) + ((valueB[1] * valueB[1]) * weight2) + ((valueB[2] * valueB[2]) * weight3)};
    const float result{0.299f * r2 + 0.587f * g2 + 0.114f * b2};
    o[pixIdx] = result > threshold ? MAX_UC : ZERO_UC;
}

__global__ void calcWeightedVarianceColor(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                      uint8_t *const o, float weight1, float weight2, float weight3)
{
    size_t pixIdx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t pixIdx3{pixIdx * 3};

    const float dI1[] = {(float)i1[pixIdx3], (float)i1[pixIdx3] + 1, (float)i1[pixIdx3 + 2]};
    const float dI2[] = {(float)i2[pixIdx3], (float)i2[pixIdx3] + 1, (float)i2[pixIdx3 + 2]};
    const float dI3[] = {(float)i3[pixIdx3], (float)i3[pixIdx3] + 1, (float)i3[pixIdx3 + 2]};
    const float meanR{(dI1[0] * weight1) + (dI2[0] * weight2) + (dI3[0] * weight3)};
    const float meanG{(dI1[1] * weight1) + (dI2[1] * weight2) + (dI3[1] * weight3)};
    const float meanB{(dI1[2] * weight1) + (dI2[2] * weight2) + (dI3[2] * weight3)};
    const float valueR[] = {dI1[0] - meanR, dI2[0] - meanR, dI2[0] - meanR};
    const float valueG[] = {dI1[1] - meanG, dI2[1] - meanG, dI2[1] - meanG};
    const float valueB[] = {dI1[2] - meanB, dI2[2] - meanB, dI2[2] - meanB};
    const float r{sqrt(((valueR[0] * valueR[0]) * weight1) + ((valueR[1] * valueR[1]) * weight2) + ((valueR[2] * valueR[2]) * weight3))};
    const float g{sqrt(((valueG[0] * valueG[0]) * weight1) + ((valueG[1] * valueG[1]) * weight2) + ((valueG[2] * valueG[2]) * weight3))};
    const float b{sqrt(((valueB[0] * valueB[0]) * weight1) + ((valueB[1] * valueB[1]) * weight2) + ((valueB[2] * valueB[2]) * weight3))};
    o[pixIdx] = 0.299f * r + 0.587f * g + 0.114f * b;
}

extern "C" void weightedVarianceColorCuda(
    const uint8_t* const img1,
    const uint8_t* const img2,
    const uint8_t* const img3,
    uint8_t* const outImg,
    const size_t numPixels,
    const WeightedMovingVarianceParams &_params)
{
    const size_t numThreads{1024};
    const size_t numBlocks{numPixels / numThreads};
    //printf("%f\n", _params.weight1);//.weight1, _params.weight2, _params.weight3);

    if (_params.enableThreshold)
        calcWeightedVarianceColorThreshold<<<numBlocks, numThreads>>>(img1, img2, img3, outImg, 
            _params.weight[0], _params.weight[1], _params.weight[2], _params.thresholdSquared);
    else
        calcWeightedVarianceColor<<<numBlocks, numThreads>>>(img1, img2, img3, outImg, 
            _params.weight[0], _params.weight[1], _params.weight[2]);

    cudaThreadSynchronize();
}