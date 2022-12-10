#include "WeightedMovingVarianceCuda.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>
#include <cuda_runtime.h>

using namespace sky360lib::bgs;

WeightedMovingVarianceCuda::WeightedMovingVarianceCuda(bool _enableWeight,
                                                       bool _enableThreshold,
                                                       float _threshold,
                                                       size_t _numProcessesParallel)
    : CoreBgs(_numProcessesParallel),
      m_params(_enableWeight, _enableThreshold, _threshold,
               _enableWeight ? DEFAULT_WEIGHTS[0] : ONE_THIRD,
               _enableWeight ? DEFAULT_WEIGHTS[1] : ONE_THIRD,
               _enableWeight ? DEFAULT_WEIGHTS[2] : ONE_THIRD)
{
}

WeightedMovingVarianceCuda::~WeightedMovingVarianceCuda()
{
    clearCuda();
}

void WeightedMovingVarianceCuda::getBackgroundImage(cv::Mat &)
{
}

void WeightedMovingVarianceCuda::clearCuda()
{
    for (size_t i = 0; i < m_numProcessesParallel; ++i)
    {
        if (imgInputPrev[i].pImgOutputCuda != nullptr)
            cudaFree(imgInputPrev[i].pImgOutputCuda);
        if (imgInputPrev[i].pImgMem[0] != nullptr)
            cudaFree(imgInputPrev[i].pImgMem[0]);
        if (imgInputPrev[i].pImgMem[1] != nullptr)
            cudaFree(imgInputPrev[i].pImgMem[1]);
        if (imgInputPrev[i].pImgMem[2] != nullptr)
            cudaFree(imgInputPrev[i].pImgMem[2]);
    }
}

void WeightedMovingVarianceCuda::initialize(const cv::Mat &)
{
    imgInputPrev.resize(m_numProcessesParallel);
    for (size_t i = 0; i < m_numProcessesParallel; ++i)
    {
        imgInputPrev[i].currentRollingIdx = 0;
        imgInputPrev[i].firstPhase = 0;
        imgInputPrev[i].pImgSize = m_imgSizesParallel[i].get();
        imgInputPrev[i].pImgInput = nullptr;
        imgInputPrev[i].pImgInputPrev1 = nullptr;
        imgInputPrev[i].pImgInputPrev2 = nullptr;
        cudaMalloc((void **)&imgInputPrev[i].pImgOutputCuda, imgInputPrev[i].pImgSize->numPixels);
        cudaMalloc((void **)&imgInputPrev[i].pImgMem[0], imgInputPrev[i].pImgSize->size);
        cudaMalloc((void **)&imgInputPrev[i].pImgMem[1], imgInputPrev[i].pImgSize->size);
        cudaMalloc((void **)&imgInputPrev[i].pImgMem[2], imgInputPrev[i].pImgSize->size);
        rollImages(imgInputPrev[i]);
    }
}

void WeightedMovingVarianceCuda::rollImages(RollingImages &rollingImages)
{
    const auto rollingIdx = ROLLING_BG_IDX[rollingImages.currentRollingIdx % 3];
    rollingImages.pImgInput = rollingImages.pImgMem[rollingIdx[0]];
    rollingImages.pImgInputPrev1 = rollingImages.pImgMem[rollingIdx[1]];
    rollingImages.pImgInputPrev2 = rollingImages.pImgMem[rollingIdx[2]];

    ++rollingImages.currentRollingIdx;
}

void WeightedMovingVarianceCuda::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput, int _numProcess)
{
    if (_imgOutput.empty())
    {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }
    process(_imgInput, _imgOutput, imgInputPrev[_numProcess], m_params);
    rollImages(imgInputPrev[_numProcess]);
}

extern "C" void weightedVarianceMonoCuda(
    const uint8_t *const img1,
    const uint8_t *const img2,
    const uint8_t *const img3,
    uint8_t *const outImg,
    const size_t numPixels,
    const WeightedMovingVarianceParams &_params);
extern "C" void weightedVarianceColorCuda(
    const uint8_t *const img1,
    const uint8_t *const img2,
    const uint8_t *const img3,
    uint8_t *const outImg,
    const size_t numPixels,
    const WeightedMovingVarianceParams &_params);

void WeightedMovingVarianceCuda::process(const cv::Mat &_imgInput,
                                         cv::Mat &_imgOutput,
                                         RollingImages &_imgInputPrev,
                                         const WeightedMovingVarianceParams &_params)
{
    const size_t numPixels = _imgInput.size().area();
    cudaMemcpyAsync(_imgInputPrev.pImgInput, _imgInput.data, numPixels * _imgInput.channels(), cudaMemcpyHostToDevice);

    if (_imgInputPrev.firstPhase < 2)
    {
        ++_imgInputPrev.firstPhase;
        return;
    }

    if (_imgInputPrev.pImgSize->numBytesPerPixel == 1)
        weightedVarianceMonoCuda(_imgInputPrev.pImgInput, _imgInputPrev.pImgInputPrev1, _imgInputPrev.pImgInputPrev2,
                                 _imgInputPrev.pImgOutputCuda, numPixels, _params);
    else
        weightedVarianceColorCuda(_imgInputPrev.pImgInput, _imgInputPrev.pImgInputPrev1, _imgInputPrev.pImgInputPrev2,
                                  _imgInputPrev.pImgOutputCuda, numPixels, _params);

    cudaMemcpyAsync(_imgOutput.data, _imgInputPrev.pImgOutputCuda, numPixels, cudaMemcpyDeviceToHost);
}
