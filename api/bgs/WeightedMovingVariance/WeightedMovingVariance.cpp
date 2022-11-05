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
    imgInputPrevParallel.resize(m_numProcessesParallel);

    for (int i = 0; i < m_numProcessesParallel; ++i)
    {
        imgInputPrevParallel[i][0] = std::make_unique<cv::Mat>();
        imgInputPrevParallel[i][1] = std::make_unique<cv::Mat>();
    }
}

WeightedMovingVariance::~WeightedMovingVariance()
{
}

void WeightedMovingVariance::getBackgroundImage(cv::Mat &_bgImage)
{
}

void WeightedMovingVariance::initialize(const cv::Mat &_image)
{
}

void WeightedMovingVariance::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput, int _numProcess)
{
    if (_imgOutput.empty())
    {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }
    process(_imgInput, _imgOutput, imgInputPrevParallel[_numProcess], m_params);
}

inline void calcWeightedVarianceMono(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                     uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI[] = {(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * _params.weight1) + (dI[1] * _params.weight2) + (dI[2] * _params.weight3)};
    const float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    *o = std::sqrt(((value[0] * value[0]) * _params.weight1) + ((value[1] * value[1]) * _params.weight2) + ((value[2] * value[2]) * _params.weight3));
}

inline void calcWeightedVarianceMonoThreshold(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                     uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI[] = {(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * _params.weight1) + (dI[1] * _params.weight2) + (dI[2] * _params.weight3)};
    const float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    const float result{((value[0] * value[0]) * _params.weight1) + ((value[1] * value[1]) * _params.weight2) + ((value[2] * value[2]) * _params.weight3)};
    *o = result > _params.thresholdSquared ? MAX_UC : ZERO_UC;
}

inline void calcWeightedVarianceColor(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                      uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI1[] = {(float)*i1, (float)*i1 + 1, (float)*i1 + 2};
    const float dI2[] = {(float)*i2, (float)*i2 + 1, (float)*i2 + 2};
    const float dI3[] = {(float)*i3, (float)*i3 + 1, (float)*i3 + 2};
    const float meanR{(dI1[0] * _params.weight1) + (dI2[0] * _params.weight2) + (dI3[0] * _params.weight3)};
    const float meanG{(dI1[1] * _params.weight1) + (dI2[1] * _params.weight2) + (dI3[1] * _params.weight3)};
    const float meanB{(dI1[2] * _params.weight1) + (dI2[2] * _params.weight2) + (dI3[2] * _params.weight3)};
    const float valueR[] = {dI1[0] - meanR, dI2[0] - meanR, dI2[0] - meanR};
    const float valueG[] = {dI1[1] - meanG, dI2[1] - meanG, dI2[1] - meanG};
    const float valueB[] = {dI1[2] - meanB, dI2[2] - meanB, dI2[2] - meanB};
    const float r{std::sqrt(((valueR[0] * valueR[0]) * _params.weight1) + ((valueR[1] * valueR[1]) * _params.weight2) + ((valueR[2] * valueR[2]) * _params.weight3))};
    const float g{std::sqrt(((valueG[0] * valueG[0]) * _params.weight1) + ((valueG[1] * valueG[1]) * _params.weight2) + ((valueG[2] * valueG[2]) * _params.weight3))};
    const float b{std::sqrt(((valueB[0] * valueB[0]) * _params.weight1) + ((valueB[1] * valueB[1]) * _params.weight2) + ((valueB[2] * valueB[2]) * _params.weight3))};
    *o = 0.299f * r + 0.587f * g + 0.114f * b;
}

inline void calcWeightedVarianceColorThreshold(const uint8_t *const i1, const uint8_t *const i2, const uint8_t *const i3,
                                      uint8_t *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI1[] = {(float)*i1, (float)*i1 + 1, (float)*i1 + 2};
    const float dI2[] = {(float)*i2, (float)*i2 + 1, (float)*i2 + 2};
    const float dI3[] = {(float)*i3, (float)*i3 + 1, (float)*i3 + 2};
    const float meanR{(dI1[0] * _params.weight1) + (dI2[0] * _params.weight2) + (dI3[0] * _params.weight3)};
    const float meanG{(dI1[1] * _params.weight1) + (dI2[1] * _params.weight2) + (dI3[1] * _params.weight3)};
    const float meanB{(dI1[2] * _params.weight1) + (dI2[2] * _params.weight2) + (dI3[2] * _params.weight3)};
    const float valueR[] = {dI1[0] - meanR, dI2[0] - meanR, dI2[0] - meanR};
    const float valueG[] = {dI1[1] - meanG, dI2[1] - meanG, dI2[1] - meanG};
    const float valueB[] = {dI1[2] - meanB, dI2[2] - meanB, dI2[2] - meanB};
    const float r2{((valueR[0] * valueR[0]) * _params.weight1) + ((valueR[1] * valueR[1]) * _params.weight2) + ((valueR[2] * valueR[2]) * _params.weight3)};
    const float g2{((valueG[0] * valueG[0]) * _params.weight1) + ((valueG[1] * valueG[1]) * _params.weight2) + ((valueG[2] * valueG[2]) * _params.weight3)};
    const float b2{((valueB[0] * valueB[0]) * _params.weight1) + ((valueB[1] * valueB[1]) * _params.weight2) + ((valueB[2] * valueB[2]) * _params.weight3)};
    const float result{0.299f * r2 + 0.587f * g2 + 0.114f * b2};
    *o = (uint8_t)(result > _params.thresholdSquared ? 255.0f : 0.0f);
}

void WeightedMovingVariance::process(const cv::Mat &_inImage,
                                     cv::Mat &_outImg,
                                     std::array<std::unique_ptr<cv::Mat>, 2> &_imgInputPrev,
                                     const WeightedMovingVarianceParams &_params)
{
    const ImgSize sizeImg{_inImage.size().width, _inImage.size().height, _inImage.channels()};

    auto inImageCopy = std::make_unique<cv::Mat>();
    _inImage.copyTo(*inImageCopy);

    if (_imgInputPrev[0]->empty())
    {
        _imgInputPrev[0] = std::move(inImageCopy);
        return;
    }

    if (_imgInputPrev[1]->empty())
    {
        _imgInputPrev[1] = std::move(_imgInputPrev[0]);
        _imgInputPrev[0] = std::move(inImageCopy);
        return;
    }

    if (sizeImg.numBytesPerPixel == 1)
        weightedVarianceMono(inImageCopy->data, _imgInputPrev[0]->data, _imgInputPrev[1]->data, 
                            _outImg.data, (size_t)sizeImg.numPixels, _params);
    else
        weightedVarianceColor(inImageCopy->data, _imgInputPrev[0]->data, _imgInputPrev[1]->data, 
                            _outImg.data, (size_t)sizeImg.numPixels, _params);

    _imgInputPrev[1] = std::move(_imgInputPrev[0]);
    _imgInputPrev[0] = std::move(inImageCopy);
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
