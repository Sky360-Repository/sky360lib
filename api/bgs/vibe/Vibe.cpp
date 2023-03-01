#include "Vibe.hpp"

#include <iostream>
#include <execution>

using namespace sky360lib::bgs;

Vibe::Vibe(VibeParams _params, size_t _numProcessesParallel)
    : CoreBgs(_numProcessesParallel)
    , m_params{_params}
{
}

void Vibe::initialize(const cv::Mat &_initImg)
{
    std::vector<std::unique_ptr<Img>> imgSplit(m_numProcessesParallel);
    m_origImgSize = ImgSize::create(_initImg.size().width, _initImg.size().height, _initImg.channels(), _initImg.elemSize1(), 0);
    Img frameImg(_initImg.data, *m_origImgSize);
    splitImg(frameImg, imgSplit, m_numProcessesParallel);

    m_randomGenerators.resize(m_numProcessesParallel);
    m_bgImgSamples.resize(m_numProcessesParallel);
    if (m_origImgSize->bytesPerPixel == 1)
    {
        for (size_t i{0}; i < m_numProcessesParallel; ++i)
        {
            initialize<uint8_t>(*imgSplit[i], m_bgImgSamples[i], m_randomGenerators[i]);
        }
    }
    else
    {
        for (size_t i{0}; i < m_numProcessesParallel; ++i)
        {
            initialize<uint16_t>(*imgSplit[i], m_bgImgSamples[i], m_randomGenerators[i]);
        }
    }
}

template<class T>
void Vibe::initialize(const Img &_initImg, std::vector<std::unique_ptr<Img>> &_bgImgSamples, Pcg32 &_rndGen)
{
    int ySample, xSample;
    _bgImgSamples.resize(m_params.NBGSamples);
    for (size_t s{0}; s < m_params.NBGSamples; ++s)
    {
        _bgImgSamples[s] = Img::create(_initImg.size, false);
        for (int yOrig{0}; yOrig < _initImg.size.height; yOrig++)
        {
            for (int xOrig{0}; xOrig < _initImg.size.width; xOrig++)
            {
                getSamplePosition_7x7_std2(_rndGen.fast(), xSample, ySample, xOrig, yOrig, _initImg.size);
                const size_t pixelPos = (yOrig * _initImg.size.width + xOrig) * _initImg.size.numChannels;
                const size_t samplePos = (ySample * _initImg.size.width + xSample) * _initImg.size.numChannels;
                _bgImgSamples[s]->ptr<T>()[pixelPos] = _initImg.ptr<T>()[samplePos];
                if (_initImg.size.numChannels > 1)
                {
                    _bgImgSamples[s]->ptr<T>()[pixelPos + 1] = _initImg.ptr<T>()[samplePos + 1];
                    _bgImgSamples[s]->ptr<T>()[pixelPos + 2] = _initImg.ptr<T>()[samplePos + 2];
                }
            }
        }
    }
}

void Vibe::process(const cv::Mat &_image, cv::Mat &_fgmask, int _numProcess)
{
    Img imgSplit(_image.data, ImgSize(_image.size().width, _image.size().height, _image.channels(), _image.elemSize1(), 0));
    Img maskPartial(_fgmask.data, ImgSize(_image.size().width, _image.size().height, _fgmask.channels(), _fgmask.elemSize1(), 0));
    if (imgSplit.size.numChannels > 1)
    {
        if (imgSplit.size.bytesPerPixel == 1)
        {
            apply3<uint8_t>(imgSplit, m_bgImgSamples[_numProcess], maskPartial, m_params, m_randomGenerators[_numProcess]);
        }
        else
        {
            apply3<uint16_t>(imgSplit, m_bgImgSamples[_numProcess], maskPartial, m_params, m_randomGenerators[_numProcess]);
        }
    }
    else
    {
        if (imgSplit.size.bytesPerPixel == 1)
        {
            apply1<uint8_t>(imgSplit, m_bgImgSamples[_numProcess], maskPartial, m_params, m_randomGenerators[_numProcess]);
        }
        else
        {
            apply1<uint16_t>(imgSplit, m_bgImgSamples[_numProcess], maskPartial, m_params, m_randomGenerators[_numProcess]);
        }
    }
}

template<class T>
void Vibe::apply3(const Img &_image,
                  std::vector<std::unique_ptr<Img>> &_bgImg,
                  Img &_fgmask,
                  const VibeParams &_params,
                  Pcg32 &_rndGen)
{
    _fgmask.clear();

    const int32_t nColorDistThreshold = sizeof(T) == 1 ? _params.NColorDistThresholdColorSquared : _params.NColorDistThresholdColor16Squared;

    size_t pixOffset{0}, colorPixOffset{0};
    for (int y{0}; y < _image.size.height; ++y)
    {
        for (int x{0}; x < _image.size.width; ++x, ++pixOffset, colorPixOffset += _image.size.numChannels)
        {
            size_t nGoodSamplesCount{0},
                nSampleIdx{0};

            const T *const pixData{&_image.ptr<T>()[colorPixOffset]};

            while (nSampleIdx < _params.NBGSamples)
            {
                const T *const bg{&_bgImg[nSampleIdx]->ptr<T>()[colorPixOffset]};
                if (L2dist3Squared(pixData, bg) < nColorDistThreshold)
                {
                    ++nGoodSamplesCount;
                    if (nGoodSamplesCount >= _params.NRequiredBGSamples)
                    {
                        break;
                    }
                }
                ++nSampleIdx;
            }
            if (nGoodSamplesCount < _params.NRequiredBGSamples)
            {
                _fgmask.data[pixOffset] = UCHAR_MAX;
            }
            else
            {
                if ((_rndGen.fast() & _params.ANDlearningRate) == 0)
                {
                    T *const bgImgPixData{&_bgImg[_rndGen.fast() & _params.ANDlearningRate]->ptr<T>()[colorPixOffset]};
                    bgImgPixData[0] = pixData[0];
                    bgImgPixData[1] = pixData[1];
                    bgImgPixData[2] = pixData[2];
                }
                if ((_rndGen.fast() & _params.ANDlearningRate) == 0)
                {
                    const int neighData{getNeighborPosition_3x3(x, y, _image.size, _rndGen.fast()) * 3};
                    T *const xyRandData{&_bgImg[_rndGen.fast() & _params.ANDlearningRate]->ptr<T>()[neighData]};
                    xyRandData[0] = pixData[0];
                    xyRandData[1] = pixData[1];
                    xyRandData[2] = pixData[2];
                }
            }
        }
    }
}

template<class T>
void Vibe::apply1(const Img &_image,
                  std::vector<std::unique_ptr<Img>> &_bgImg,
                  Img &_fgmask,
                  const VibeParams &_params,
                  Pcg32 &_rndGen)
{
    _fgmask.clear();

    const int32_t nColorDistThreshold = sizeof(T) == 1 ? _params.NColorDistThresholdMono : _params.NColorDistThresholdMono16;

    size_t pixOffset{0};
    for (int y{0}; y < _image.size.height; ++y)
    {
        for (int x{0}; x < _image.size.width; ++x, ++pixOffset)
        {
            uint32_t nGoodSamplesCount{0},
                nSampleIdx{0};

            const T pixData{_image.ptr<T>()[pixOffset]};

            while (nSampleIdx < _params.NBGSamples)
            {
                if (std::abs((int32_t)_bgImg[nSampleIdx]->ptr<T>()[pixOffset] - (int32_t)pixData) < nColorDistThreshold)
                {
                    ++nGoodSamplesCount;
                    if (nGoodSamplesCount >= _params.NRequiredBGSamples)
                    {
                        break;
                    }
                }
                ++nSampleIdx;
            }
            if (nGoodSamplesCount < _params.NRequiredBGSamples)
            {
                _fgmask.data[pixOffset] = UCHAR_MAX;
            }
            else
            {
                if ((_rndGen.fast() & _params.ANDlearningRate) == 0)
                {
                    _bgImg[_rndGen.fast() & _params.ANDlearningRate]->ptr<T>()[pixOffset] = pixData;
                }
                if ((_rndGen.fast() & _params.ANDlearningRate) == 0)
                {
                    const int neighData{getNeighborPosition_3x3(x, y, _image.size, _rndGen.fast())};
                    _bgImg[_rndGen.fast() & _params.ANDlearningRate]->ptr<T>()[neighData] = pixData;
                }
            }
        }
    }
}

void Vibe::getBackgroundImage(cv::Mat &backgroundImage)
{
    cv::Mat oAvgBGImg(m_origImgSize->height, m_origImgSize->width, CV_32FC(m_origImgSize->numChannels));

    for (size_t t{0}; t < m_numProcessesParallel; ++t)
    {
        const std::vector<std::unique_ptr<Img>> &bgSamples = m_bgImgSamples[t];
        for (size_t n{0}; n < m_params.NBGSamples; ++n)
        {
            size_t inPixOffset{0};
            size_t outPixOffset{bgSamples[0]->size.originalPixelPos * sizeof(float) * bgSamples[0]->size.numChannels};
            for (; inPixOffset < bgSamples[n]->size.sizeInBytes;
                 inPixOffset += m_origImgSize->numChannels,
                 outPixOffset += sizeof(float) * bgSamples[0]->size.numChannels)
            {
                const uint8_t *const pixData{&bgSamples[n]->data[inPixOffset]};
                float *const outData{(float *)(oAvgBGImg.data + outPixOffset)};
                for (int c{0}; c < m_origImgSize->numChannels; ++c)
                {
                    outData[c] += (float)pixData[c] / (float)m_params.NBGSamples;
                }
            }
        }
    }

    oAvgBGImg.convertTo(backgroundImage, CV_8U);
}
