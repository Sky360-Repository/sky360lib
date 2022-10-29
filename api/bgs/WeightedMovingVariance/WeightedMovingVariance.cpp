#include "WeightedMovingVariance.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

using namespace bgslibrary::algorithms;

WeightedMovingVariance::WeightedMovingVariance(bool _enableWeight,
                                    bool _enableThreshold,
                                    int _threshold) 
    : m_params(_enableWeight, _enableThreshold, _threshold),
    m_numProcessesParallel(12)
{
    imgInputPrevParallel.resize(m_numProcessesParallel);

    for (int i = 0; i < m_numProcessesParallel; ++i) {
        m_processSeq.push_back(i);
        imgInputPrevParallel[i][0] = std::make_unique<cv::Mat>();
        imgInputPrevParallel[i][1] = std::make_unique<cv::Mat>();
    }
}

WeightedMovingVariance::~WeightedMovingVariance()
{
}

void WeightedMovingVariance::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput)
{
    if (_imgOutput.empty()) {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }
    if (m_numProcessesParallel > 1) 
        processParallel(_imgInput, _imgOutput);
    else
        process(_imgInput, _imgOutput, imgInputPrevParallel[0], m_params);
}

void WeightedMovingVariance::processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput) {
    std::for_each(
        std::execution::par,
        m_processSeq.begin(),
        m_processSeq.end(),
        [&](int np)
        {
            const int height{_imgInput.size().height / m_numProcessesParallel};
            const int pixelPos{np * _imgInput.size().width * height};
            cv::Mat imgSplit(height, _imgInput.size().width, _imgInput.type(), _imgInput.data + (pixelPos * _imgInput.channels()));
            cv::Mat maskPartial(height, _imgInput.size().width, _imgOutput.type(), _imgOutput.data + pixelPos);
            process(imgSplit, maskPartial, imgInputPrevParallel[np], m_params);
        });
}

void WeightedMovingVariance::process(const cv::Mat &_inImage, 
                                    cv::Mat &_outImg, 
                                    std::array<std::unique_ptr<cv::Mat>, 2>& _imgInputPrev, 
                                    const WeightedMovingVarianceParams& _params)
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

inline float calcWeightedVariance(const uchar* const i1, const uchar* const i2, const uchar* const i3,
                        const float w1, const float w2, const float w3) {
    const float dI[] = {(float)*i1, (float)*i2, (float)*i3};
    const float mean{(dI[0] * w1) + (dI[1] * w2) + (dI[2] * w3)};
    const float value[] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    return std::sqrt(((value[0] * value[0]) * w1) + ((value[1] * value[1]) * w2) + ((value[2] * value[2]) * w3));
} 

void WeightedMovingVariance::weightedVarianceMono(
        const cv::Mat &img1, 
        const cv::Mat &img2, 
        const cv::Mat &img3, 
        cv::Mat& outImg,
        const WeightedMovingVarianceParams& _params)
{
    const float weight1{_params.enableWeight ? 0.5f : ONE_THIRD}; 
    const float weight2{_params.enableWeight ? 0.3f : ONE_THIRD}; 
    const float weight3{_params.enableWeight ? 0.2f : ONE_THIRD}; 

    size_t totalDataSize{(size_t)img1.size().area()};
    uchar *dataI1{img1.data};
    uchar *dataI2{img2.data};
    uchar *dataI3{img3.data};
    uchar *dataOut{outImg.data};
    for (size_t i{0}; i < totalDataSize; ++i) {
        float result{calcWeightedVariance(dataI1, dataI2, dataI3, weight1, weight2, weight3)};
        *dataOut = _params.enableThreshold ? ((uchar)(result > _params.threshold ? 255 : 0))
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
        cv::Mat& outImg,
        const WeightedMovingVarianceParams& _params)
{
    const float weight1{_params.enableWeight ? 0.5f : ONE_THIRD}; 
    const float weight2{_params.enableWeight ? 0.3f : ONE_THIRD}; 
    const float weight3{_params.enableWeight ? 0.2f : ONE_THIRD}; 
    const int numChannels = img1.channels();

    size_t totalDataSize{(size_t)img1.size().area()};
    uchar *dataI1{img1.data};
    uchar *dataI2{img2.data};
    uchar *dataI3{img3.data};
    uchar *dataOut{outImg.data};
    for (size_t i{0}; i < totalDataSize; ++i) {
        const float r{calcWeightedVariance(dataI1, dataI2, dataI3, weight1, weight2, weight3)};
        const float g{calcWeightedVariance(dataI1 + 1, dataI2 + 1, dataI3 + 1, weight1, weight2, weight3)};
        const float b{calcWeightedVariance(dataI1 + 2, dataI2 + 2, dataI3 + 2, weight1, weight2, weight3)};
        const float result{0.299f * r + 0.587f * g + 0.114f * b};
        *dataOut = _params.enableThreshold ? ((uchar)(result > _params.threshold ? 255 : 0))
                                            : (uchar)result;
        ++dataOut;
        dataI1 += numChannels;
        dataI2 += numChannels;
        dataI3 += numChannels;
    }
}
