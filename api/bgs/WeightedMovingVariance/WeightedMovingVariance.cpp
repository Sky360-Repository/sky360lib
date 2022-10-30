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
        weightedVarianceMono(*inImageCopy, *_imgInputPrev[0], *_imgInputPrev[1], _outImg, _params);
    else
        weightedVarianceColor(*inImageCopy, *_imgInputPrev[0], *_imgInputPrev[1], _outImg, _params);

    _imgInputPrev[1] = std::move(_imgInputPrev[0]);
    _imgInputPrev[0] = std::move(inImageCopy);
}

inline float calcWeightedVariance(const uchar *const i1, const uchar *const i2, const uchar *const i3,
                                  const WeightedMovingVarianceParams &_params)
{
    const float dI[] = {(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * _params.weight1) + (dI[1] * _params.weight2) + (dI[2] * _params.weight3)};
    const float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    return std::sqrt(((value[0] * value[0]) * _params.weight1) + ((value[1] * value[1]) * _params.weight2) + ((value[2] * value[2]) * _params.weight3));
}

void WeightedMovingVariance::weightedVarianceMono(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const cv::Mat &img3,
    cv::Mat &outImg,
    const WeightedMovingVarianceParams &_params)
{
    size_t totalDataSize{(size_t)img1.size().area()};
    uchar *dataI1{img1.data};
    uchar *dataI2{img2.data};
    uchar *dataI3{img3.data};
    uchar *dataOut{outImg.data};
    for (size_t i{0}; i < totalDataSize; ++i)
    {
        float result{calcWeightedVariance(dataI1, dataI2, dataI3, _params)};
        *dataOut = _params.enableThreshold ? ((uchar)(result > _params.threshold ? 255.0f : 0.0f))
                                           : (uchar)result;
        ++dataOut;
        ++dataI1;
        ++dataI2;
        ++dataI3;
    }
}

void WeightedMovingVariance::weightedVarianceColor(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const cv::Mat &img3,
    cv::Mat &outImg,
    const WeightedMovingVarianceParams &_params)
{
    const int numChannels = img1.channels();

    size_t totalDataSize{(size_t)img1.size().area()};
    uchar *dataI1{img1.data};
    uchar *dataI2{img2.data};
    uchar *dataI3{img3.data};
    uchar *dataOut{outImg.data};
    for (size_t i{0}; i < totalDataSize; ++i)
    {
        const float r{calcWeightedVariance(dataI1, dataI2, dataI3, _params)};
        const float g{calcWeightedVariance(dataI1 + 1, dataI2 + 1, dataI3 + 1, _params)};
        const float b{calcWeightedVariance(dataI1 + 2, dataI2 + 2, dataI3 + 2, _params)};
        const float result{0.299f * r + 0.587f * g + 0.114f * b};
        *dataOut = _params.enableThreshold ? ((uchar)(result > _params.threshold ? 255.0f : 0.0f))
                                           : (uchar)result;
        ++dataOut;
        dataI1 += numChannels;
        dataI2 += numChannels;
        dataI3 += numChannels;
    }
}
