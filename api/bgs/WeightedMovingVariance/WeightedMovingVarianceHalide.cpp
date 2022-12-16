#include "WeightedMovingVarianceHalide.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

#include "wmv_mono_auto_schedule.h"
#include "wmv_mono_threshold_auto_schedule.h"
#include "wmv_color_auto_schedule.h"
#include "wmv_color_threshold_auto_schedule.h"

#include <HalideBuffer.h>

using namespace sky360lib::bgs;
using namespace Halide::Runtime;

WeightedMovingVarianceHalide::WeightedMovingVarianceHalide(bool _enableWeight,
                                               bool _enableThreshold,
                                               float _threshold)
    : CoreBgs(1),
      m_params(_enableWeight, _enableThreshold, _threshold,
               _enableWeight ? DEFAULT_WEIGHTS[0] : ONE_THIRD,
               _enableWeight ? DEFAULT_WEIGHTS[1] : ONE_THIRD,
               _enableWeight ? DEFAULT_WEIGHTS[2] : ONE_THIRD)
{
}

WeightedMovingVarianceHalide::~WeightedMovingVarianceHalide()
{
}

void WeightedMovingVarianceHalide::getBackgroundImage(cv::Mat &)
{
}

void WeightedMovingVarianceHalide::initialize(const cv::Mat &)
{
    initParallelData();
}

void WeightedMovingVarianceHalide::initParallelData()
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
        imgInputPrev[i].pImgMem[0] = std::make_unique_for_overwrite<uint8_t[]>(imgInputPrev[i].pImgSize->size);
        imgInputPrev[i].pImgMem[1] = std::make_unique_for_overwrite<uint8_t[]>(imgInputPrev[i].pImgSize->size);
        imgInputPrev[i].pImgMem[2] = std::make_unique_for_overwrite<uint8_t[]>(imgInputPrev[i].pImgSize->size);
        rollImages(imgInputPrev[i]);
    }
}

void WeightedMovingVarianceHalide::rollImages(RollingImages& rollingImages)
{
    const auto rollingIdx = ROLLING_BG_IDX[rollingImages.currentRollingIdx % 3];
    rollingImages.pImgInput = rollingImages.pImgMem[rollingIdx[0]].get();
    rollingImages.pImgInputPrev1 = rollingImages.pImgMem[rollingIdx[1]].get();
    rollingImages.pImgInputPrev2 = rollingImages.pImgMem[rollingIdx[2]].get();

    ++rollingImages.currentRollingIdx;
}

void WeightedMovingVarianceHalide::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput, int _numProcess)
{
    if (_imgOutput.empty())
    {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }
    process(_imgInput, _imgOutput, imgInputPrev[_numProcess], m_params);
    rollImages(imgInputPrev[_numProcess]);
}

void WeightedMovingVarianceHalide::process(const cv::Mat &_inImage,
                                     cv::Mat &_outImg,
                                     RollingImages &_imgInputPrev,
                                     const WeightedMovingVarianceParams &_params)
{
    memcpy(_imgInputPrev.pImgInput, _inImage.data, _imgInputPrev.pImgSize->size);

    if (_imgInputPrev.firstPhase < 2)
    {
        ++_imgInputPrev.firstPhase;
        return;
    }

    if (_imgInputPrev.pImgSize->numBytesPerPixel == 1)
        weightedVarianceMono(_imgInputPrev.pImgInput, _imgInputPrev.pImgInputPrev1, _imgInputPrev.pImgInputPrev2, 
                            _outImg.data, _imgInputPrev.pImgSize->width, _imgInputPrev.pImgSize->height, _params);
    else
        weightedVarianceColor(_imgInputPrev.pImgInput, _imgInputPrev.pImgInputPrev1, _imgInputPrev.pImgInputPrev2, 
                            _outImg.data, _imgInputPrev.pImgSize->width, _imgInputPrev.pImgSize->height, _params);
}

void WeightedMovingVarianceHalide::weightedVarianceMono(
    uint8_t *const img1,
    uint8_t *const img2,
    uint8_t *const img3,
    uint8_t *const outImg,
    const int width,
    const int height,
    const WeightedMovingVarianceParams &_params)
{
    Buffer<uint8_t> input0(img1, width, height);
    Buffer<uint8_t> input1(img2, width, height);
    Buffer<uint8_t> input2(img3, width, height);
    Buffer<uint8_t> output(outImg, width, height);

    if (_params.enableThreshold)
        wmv_mono_threshold_auto_schedule(input0, input1, input2, _params.weight[0], _params.weight[1], _params.weight[2], _params.thresholdSquared, output);
    else
        wmv_mono_auto_schedule(input0, input1, input2, _params.weight[0], _params.weight[1], _params.weight[2], output);
    output.device_sync();
}

void WeightedMovingVarianceHalide::weightedVarianceColor(
    const uint8_t *const img1,
    const uint8_t *const img2,
    const uint8_t *const img3,
    uint8_t *const outImg,
    const int width,
    const int height,
    const WeightedMovingVarianceParams &_params)
{
    Buffer<uint8_t> input0{Buffer<uint8_t>::make_interleaved((uint8_t*)img1, width, height, 3)};
    Buffer<uint8_t> input1{Buffer<uint8_t>::make_interleaved((uint8_t*)img2, width, height, 3)};
    Buffer<uint8_t> input2{Buffer<uint8_t>::make_interleaved((uint8_t*)img3, width, height, 3)};
    Buffer<uint8_t> output(outImg, width, height);

    if (_params.enableThreshold)
        wmv_color_threshold_auto_schedule(input0, input1, input2, _params.weight[0], _params.weight[1], _params.weight[2], _params.thresholdSquared, output);
    else
        wmv_color_auto_schedule(input0, input1, input2, _params.weight[0], _params.weight[1], _params.weight[2], output);
    output.device_sync();
}
