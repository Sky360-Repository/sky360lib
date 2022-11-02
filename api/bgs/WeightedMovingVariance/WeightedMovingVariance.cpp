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

void WeightedMovingVariance::process(const cv::Mat &_inImage,
                                     cv::Mat &_outImg,
                                     std::array<std::unique_ptr<cv::Mat>, 2> &_imgInputPrev,
                                     const WeightedMovingVarianceParams &_params)
{
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

    if (inImageCopy->channels() == 1)
        weightedVarianceMono(inImageCopy->data, _imgInputPrev[0]->data, _imgInputPrev[1]->data, 
                            _outImg.data, (size_t)inImageCopy->size().area(), _params);
    else
        weightedVarianceColor(inImageCopy->data, _imgInputPrev[0]->data, _imgInputPrev[1]->data, 
                            _outImg.data, (size_t)inImageCopy->size().area(), _params);

    _imgInputPrev[1] = std::move(_imgInputPrev[0]);
    _imgInputPrev[0] = std::move(inImageCopy);
}

inline void calcWeightedVarianceMono(const uchar *const i1, const uchar *const i2, const uchar *const i3,
                                     uchar *const o, const WeightedMovingVarianceParams &_params)
{
    const float dI[] = {(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * _params.weight1) + (dI[1] * _params.weight2) + (dI[2] * _params.weight3)};
    const float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    const float result{std::sqrt(((value[0] * value[0]) * _params.weight1) + ((value[1] * value[1]) * _params.weight2) + ((value[2] * value[2]) * _params.weight3))};
    *o = _params.enableThreshold ? ((uchar)(result > _params.threshold ? 255.0f : 0.0f))
                                 : (uchar)result;
}

void WeightedMovingVariance::weightedVarianceMono(
    const uchar *const img1,
    const uchar *const img2,
    const uchar *const img3,
    uchar *const outImg,
    const size_t totalPixels,
    const WeightedMovingVarianceParams &_params)
{
    for (size_t i{0}; i < totalPixels; ++i)
    {
        calcWeightedVarianceMono(img1 + i, img2 + i, img3 + i, outImg + i, _params);
    }
}

inline void calcWeightedVarianceColor(const uchar *const i1, const uchar *const i2, const uchar *const i3,
                                      uchar *const o, const WeightedMovingVarianceParams &_params)
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
    const float result{0.299f * r + 0.587f * g + 0.114f * b};
    *o = _params.enableThreshold ? ((uchar)(result > _params.threshold ? 255.0f : 0.0f))
                                 : (uchar)result;
}

void WeightedMovingVariance::weightedVarianceColor(
    const uchar *const img1,
    const uchar *const img2,
    const uchar *const img3,
    uchar *const outImg,
    const size_t totalPixels,
    const WeightedMovingVarianceParams &_params)
{
    for (size_t i{0}, i3{0}; i < totalPixels; ++i, i3 += 3)
    {
        calcWeightedVarianceColor(img1 + i3, img2 + i3, img3 + i3, outImg + i, _params);
    }
}
