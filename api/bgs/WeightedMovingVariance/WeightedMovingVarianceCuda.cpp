#include "WeightedMovingVarianceCuda.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace sky360lib::bgs;

WeightedMovingVarianceCuda::WeightedMovingVarianceCuda(bool _enableWeight,
                                                       bool _enableThreshold,
                                                       float _threshold)
    : CoreBgs(1),
      m_params(_enableWeight, _enableThreshold, _threshold,
               _enableWeight ? DEFAULT_WEIGHTS[0] : ONE_THIRD,
               _enableWeight ? DEFAULT_WEIGHTS[1] : ONE_THIRD,
               _enableWeight ? DEFAULT_WEIGHTS[2] : ONE_THIRD),
      m_currentRollingIdx{0},
      m_firstPhase{0}
{
    m_pImgInputCuda = nullptr;
    m_pImgInputPrev1Cuda = nullptr;
    m_pImgInputPrev2Cuda = nullptr;

    m_pImgOutputCuda = nullptr;
    m_pImgMemCuda[0] = nullptr;
    m_pImgMemCuda[1] = nullptr;
    m_pImgMemCuda[2] = nullptr;
}

WeightedMovingVarianceCuda::~WeightedMovingVarianceCuda()
{
    clearCuda();
}

void WeightedMovingVarianceCuda::getBackgroundImage(cv::Mat &_bgImage)
{
}

void WeightedMovingVarianceCuda::clearCuda()
{
    if (m_pImgOutputCuda != nullptr)
        cudaFree(m_pImgOutputCuda);
    if (m_pImgMemCuda[0] != nullptr)
        cudaFree(m_pImgMemCuda[0]);
    if (m_pImgMemCuda[1] != nullptr)
        cudaFree(m_pImgMemCuda[1]);
    if (m_pImgMemCuda[2] != nullptr)
        cudaFree(m_pImgMemCuda[2]);
}

void WeightedMovingVarianceCuda::initialize(const cv::Mat &_image)
{
    clearCuda();

    const size_t size = _image.size().area() * _image.channels();
    m_pImgOutputCuda = (uchar *)malloc(_image.size().area());
    m_pImgMemCuda[0] = (uchar*)malloc(size);
    m_pImgMemCuda[1] = (uchar*)malloc(size);
    m_pImgMemCuda[2] = (uchar*)malloc(size);
    // cudaMalloc((void **)&m_pImgOutputCuda, _image.size().area());
    // cudaMalloc((void **)&m_pImgMemCuda[0], size);
    // cudaMalloc((void **)&m_pImgMemCuda[1], size);
    // cudaMalloc((void **)&m_pImgMemCuda[2], size);

    rollImages();
}

void WeightedMovingVarianceCuda::rollImages()
{
    const auto rollingIdx = ROLLING_BG_IDX[m_currentRollingIdx % 3];
    m_pImgInputCuda = m_pImgMemCuda[rollingIdx[0]];
    m_pImgInputPrev1Cuda = m_pImgMemCuda[rollingIdx[1]];
    m_pImgInputPrev2Cuda = m_pImgMemCuda[rollingIdx[2]];

    ++m_currentRollingIdx;
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

void WeightedMovingVarianceCuda::weightedVarianceMono(
    uchar* const img1,
    uchar* const img2,
    uchar* const img3,
    uchar* outImg,
    const size_t numPixels,
    const WeightedMovingVarianceParams &_params)
{
    uchar *dataI1{img1};
    uchar *dataI2{img2};
    uchar *dataI3{img3};
    uchar *dataOut{outImg};
    for (size_t i{0}; i < numPixels; ++i)
    {
        calcWeightedVarianceMono(dataI1, dataI2, dataI3, dataOut, _params);
        ++dataOut;
        ++dataI1;
        ++dataI2;
        ++dataI3;
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

void WeightedMovingVarianceCuda::weightedVarianceColor(
    uchar* const img1,
    uchar* const img2,
    uchar* const img3,
    uchar* outImg,
    const size_t numPixels,
    const int numChannels,
    const WeightedMovingVarianceParams &_params)
{
    uchar *dataI1{img1};
    uchar *dataI2{img2};
    uchar *dataI3{img3};
    uchar *dataOut{outImg};
    for (size_t i{0}; i < numPixels; ++i)
    {
        calcWeightedVarianceColor(dataI1, dataI2, dataI3, dataOut, _params);
        ++dataOut;
        dataI1 += numChannels;
        dataI2 += numChannels;
        dataI3 += numChannels;
    }
}

void WeightedMovingVarianceCuda::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput, int _numProcess)
{
    if (_imgOutput.empty())
    {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }

    const size_t numPixels = _imgInput.size().area();

    //cudaMemcpy(m_pImgInputCuda, _imgInput.data, numPixels * _imgInput.channels(), cudaMemcpyHostToDevice);
    memcpy(m_pImgInputCuda, _imgInput.data, numPixels * _imgInput.channels());

    if (m_firstPhase < 2)
    {
        rollImages();
        ++m_firstPhase;
        return;
    }

    if (_imgInput.channels() == 1)
        weightedVarianceMono(m_pImgInputCuda, m_pImgInputPrev1Cuda, m_pImgInputPrev1Cuda, m_pImgOutputCuda, 
                            numPixels, m_params);
    else
        weightedVarianceColor(m_pImgInputCuda, m_pImgInputPrev1Cuda, m_pImgInputPrev1Cuda, m_pImgOutputCuda, 
                            numPixels, _imgInput.channels() ,m_params);

    // cudaMemcpy(_imgOutput.data, m_pImgOutputCuda, numPixels, cudaMemcpyDeviceToHost);
    memcpy(_imgOutput.data, m_pImgOutputCuda, numPixels);

    rollImages();
}
