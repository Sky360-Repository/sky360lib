#include "WeightedMovingVariance.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

using namespace sky360lib::bgs;

WeightedMovingVariance::WeightedMovingVariance(bool _enableWeight,
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

WeightedMovingVariance::~WeightedMovingVariance()
{
}

void WeightedMovingVariance::getBackgroundImage(cv::Mat &_bgImage)
{
}

void WeightedMovingVariance::initialize(const cv::Mat &_image)
{
    imgInputPrev.resize(m_numProcessesParallel);
    for (int i = 0; i < m_numProcessesParallel; ++i)
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

void WeightedMovingVariance::rollImages(RollingImages& rollingImages)
{
    const auto rollingIdx = ROLLING_BG_IDX[rollingImages.currentRollingIdx % 3];
    rollingImages.pImgInput = rollingImages.pImgMem[rollingIdx[0]].get();
    rollingImages.pImgInputPrev1 = rollingImages.pImgMem[rollingIdx[1]].get();
    rollingImages.pImgInputPrev2 = rollingImages.pImgMem[rollingIdx[2]].get();

    ++rollingImages.currentRollingIdx;
}

void WeightedMovingVariance::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput, int _numProcess)
{
    if (_imgOutput.empty())
    {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }
    process(_imgInput, _imgOutput, imgInputPrev[_numProcess], m_params);
    rollImages(imgInputPrev[_numProcess]);
}

void WeightedMovingVariance::process(const cv::Mat &_inImage,
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
                            _outImg.data, (size_t)_imgInputPrev.pImgSize->numPixels, _params);
    else
        weightedVarianceColor(_imgInputPrev.pImgInput, _imgInputPrev.pImgInputPrev1, _imgInputPrev.pImgInputPrev2, 
                            _outImg.data, (size_t)_imgInputPrev.pImgSize->numPixels, _params);
}

inline void calcWeightedVarianceMono(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                     uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI[]{(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * _params.weight[0]) + (dI[1] * _params.weight[1]) + (dI[2] * _params.weight[2])};
    const float value[]{dI[0] - mean, dI[1] - mean, dI[2] - mean};
    *o = std::sqrt(((value[0] * value[0]) * _params.weight[0]) 
                    + ((value[1] * value[1]) * _params.weight[1]) 
                    + ((value[2] * value[2]) * _params.weight[2]));
}

inline void calcWeightedVarianceMonoThreshold(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                     uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI[]{(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * _params.weight[0]) + (dI[1] * _params.weight[1]) + (dI[2] * _params.weight[2])};
    const float value[]{dI[0] - mean, dI[1] - mean, dI[2] - mean};
    const float result{((value[0] * value[0]) * _params.weight[0]) 
                        + ((value[1] * value[1]) * _params.weight[1]) 
                        + ((value[2] * value[2]) * _params.weight[2])};
    *o = result > _params.thresholdSquared ? UCHAR_MAX : ZERO_UC;
}

inline void calcWeightedVarianceColor(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                      uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI1[]{(float)*i1, (float)*i1 + 1, (float)*i1 + 2};
    const float dI2[]{(float)*i2, (float)*i2 + 1, (float)*i2 + 2};
    const float dI3[]{(float)*i3, (float)*i3 + 1, (float)*i3 + 2};
    const float meanR{(dI1[0] * _params.weight[0]) + (dI2[0] * _params.weight[1]) + (dI3[0] * _params.weight[2])};
    const float meanG{(dI1[1] * _params.weight[0]) + (dI2[1] * _params.weight[1]) + (dI3[1] * _params.weight[2])};
    const float meanB{(dI1[2] * _params.weight[0]) + (dI2[2] * _params.weight[1]) + (dI3[2] * _params.weight[2])};
    const float valueR[]{dI1[0] - meanR, dI2[0] - meanR, dI2[0] - meanR};
    const float valueG[]{dI1[1] - meanG, dI2[1] - meanG, dI2[1] - meanG};
    const float valueB[]{dI1[2] - meanB, dI2[2] - meanB, dI2[2] - meanB};
    const float r{std::sqrt(((valueR[0] * valueR[0]) * _params.weight[0]) 
                            + ((valueR[1] * valueR[1]) * _params.weight[1]) 
                            + ((valueR[2] * valueR[2]) * _params.weight[2]))};
    const float g{std::sqrt(((valueG[0] * valueG[0]) * _params.weight[0]) 
                            + ((valueG[1] * valueG[1]) * _params.weight[1]) 
                            + ((valueG[2] * valueG[2]) * _params.weight[2]))};
    const float b{std::sqrt(((valueB[0] * valueB[0]) * _params.weight[0]) 
                            + ((valueB[1] * valueB[1]) * _params.weight[1]) 
                            + ((valueB[2] * valueB[2]) * _params.weight[2]))};
    *o = 0.299f * r + 0.587f * g + 0.114f * b;
}

inline void calcWeightedVarianceColorThreshold(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                      uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI1[]{(float)*i1, (float)*i1 + 1, (float)*i1 + 2};
    const float dI2[]{(float)*i2, (float)*i2 + 1, (float)*i2 + 2};
    const float dI3[]{(float)*i3, (float)*i3 + 1, (float)*i3 + 2};
    const float meanR{(dI1[0] * _params.weight[0]) + (dI2[0] * _params.weight[1]) + (dI3[0] * _params.weight[2])};
    const float meanG{(dI1[1] * _params.weight[0]) + (dI2[1] * _params.weight[1]) + (dI3[1] * _params.weight[2])};
    const float meanB{(dI1[2] * _params.weight[0]) + (dI2[2] * _params.weight[1]) + (dI3[2] * _params.weight[2])};
    const float valueR[]{dI1[0] - meanR, dI2[0] - meanR, dI2[0] - meanR};
    const float valueG[]{dI1[1] - meanG, dI2[1] - meanG, dI2[1] - meanG};
    const float valueB[]{dI1[2] - meanB, dI2[2] - meanB, dI2[2] - meanB};
    const float r2{((valueR[0] * valueR[0]) * _params.weight[0]) 
                    + ((valueR[1] * valueR[1]) * _params.weight[1]) 
                    + ((valueR[2] * valueR[2]) * _params.weight[2])};
    const float g2{((valueG[0] * valueG[0]) * _params.weight[0]) 
                    + ((valueG[1] * valueG[1]) * _params.weight[1]) 
                    + ((valueG[2] * valueG[2]) * _params.weight[2])};
    const float b2{((valueB[0] * valueB[0]) * _params.weight[0]) 
                    + ((valueB[1] * valueB[1]) * _params.weight[1]) 
                    + ((valueB[2] * valueB[2]) * _params.weight[2])};
    const float result{0.299f * r2 + 0.587f * g2 + 0.114f * b2};
    *o = result > _params.thresholdSquared ? UCHAR_MAX : ZERO_UC;
}

void WeightedMovingVariance::weightedVarianceMono(
    const uint8_t *const img1,
    const uint8_t *const img2,
    const uint8_t *const img3,
    uint8_t *const outImg,
    const size_t totalPixels,
    const WeightedMovingVarianceParams &_params)
{
    if (_params.enableThreshold)
        for (size_t i{0}; i < totalPixels; ++i)
            calcWeightedVarianceMonoThreshold(img1 + i, img2 + i, img3 + i, outImg + i, _params);
    else
        for (size_t i{0}; i < totalPixels; ++i)
            calcWeightedVarianceMono(img1 + i, img2 + i, img3 + i, outImg + i, _params);
}

void WeightedMovingVariance::weightedVarianceColor(
    const uint8_t *const img1,
    const uint8_t *const img2,
    const uint8_t *const img3,
    uint8_t *const outImg,
    const size_t totalPixels,
    const WeightedMovingVarianceParams &_params)
{
    if (_params.enableThreshold)
        for (size_t i{0}, i3{0}; i < totalPixels; ++i, i3 += 3)
            calcWeightedVarianceColorThreshold(img1 + i3, img2 + i3, img3 + i3, outImg + i, _params);
    else
        for (size_t i{0}, i3{0}; i < totalPixels; ++i, i3 += 3)
            calcWeightedVarianceColor(img1 + i3, img2 + i3, img3 + i3, outImg + i, _params);
}
