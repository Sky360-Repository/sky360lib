#include "Vibe.hpp"

#include <iostream>
#include <execution>

using namespace sky360lib::bgs;

Vibe::Vibe(const VibeParams &_params, size_t _numProcessesParallel)
    : CoreBgs(_numProcessesParallel), m_params(_params)
{
}

void Vibe::initialize(const cv::Mat &_initImg)
{
    std::vector<std::unique_ptr<Img>> imgSplit(m_numProcessesParallel);
    m_origImgSize = ImgSize::create(_initImg.size().width, _initImg.size().height, _initImg.channels());
    Img frameImg(_initImg.data, *m_origImgSize);
    splitImg(frameImg, imgSplit, m_numProcessesParallel);

    m_randomGenerators.resize(m_numProcessesParallel);
    m_bgImgSamples.resize(m_numProcessesParallel);
    for (size_t i{0}; i < m_numProcessesParallel; ++i)
    {
        initialize(*imgSplit[i], m_bgImgSamples[i], m_randomGenerators[i]);
    }
}

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
                const size_t pixelPos = (yOrig * _initImg.size.width + xOrig) * _initImg.size.numBytesPerPixel;
                const size_t samplePos = (ySample * _initImg.size.width + xSample) * _initImg.size.numBytesPerPixel;
                _bgImgSamples[s]->data[pixelPos] = _initImg.data[samplePos];
                if (_initImg.size.numBytesPerPixel > 1)
                {
                    _bgImgSamples[s]->data[pixelPos + 1] = _initImg.data[samplePos + 1];
                    _bgImgSamples[s]->data[pixelPos + 2] = _initImg.data[samplePos + 2];
                }
            }
        }
    }
}

void Vibe::process(const cv::Mat &_image, cv::Mat &_fgmask, int _numProcess)
{
    Img imgSplit(_image.data, ImgSize(_image.size().width, _image.size().height, _image.channels()));
    Img maskPartial(_fgmask.data, ImgSize(_image.size().width, _image.size().height, _fgmask.channels()));
    if (imgSplit.size.numBytesPerPixel > 1)
        apply3(imgSplit, m_bgImgSamples[_numProcess], maskPartial, m_params, m_randomGenerators[_numProcess]);
    else
        apply1(imgSplit, m_bgImgSamples[_numProcess], maskPartial, m_params, m_randomGenerators[_numProcess]);
}

void Vibe::apply3(const Img &_image,
                  std::vector<std::unique_ptr<Img>> &_bgImg,
                  Img &_fgmask,
                  const VibeParams &_params,
                  Pcg32 &_rndGen)
{
    _fgmask.clear();

    size_t pixOffset{0}, colorPixOffset{0};
    for (int y{0}; y < _image.size.height; ++y)
    {
        for (int x{0}; x < _image.size.width; ++x, ++pixOffset, colorPixOffset += _image.size.numBytesPerPixel)
        {
            size_t nGoodSamplesCount{0},
                nSampleIdx{0};

            const uint8_t *const pixData{&_image.data[colorPixOffset]};

            while (nSampleIdx < _params.NBGSamples)
            {
                const uint8_t *const bg{&_bgImg[nSampleIdx]->data[colorPixOffset]};
                if (L2dist3Squared(pixData, bg) < _params.NColorDistThresholdSquared)
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
                    uint8_t *const bgImgPixData{&_bgImg[_rndGen.fast() & _params.ANDlearningRate]->data[colorPixOffset]};
                    bgImgPixData[0] = pixData[0];
                    bgImgPixData[1] = pixData[1];
                    bgImgPixData[2] = pixData[2];
                }
                if ((_rndGen.fast() & _params.ANDlearningRate) == 0)
                {
                    const int neighData{getNeighborPosition_3x3(x, y, _image.size, _rndGen.fast()) * 3};
                    uint8_t *const xyRandData{&_bgImg[_rndGen.fast() & _params.ANDlearningRate]->data[neighData]};
                    xyRandData[0] = pixData[0];
                    xyRandData[1] = pixData[1];
                    xyRandData[2] = pixData[2];
                }
            }
        }
    }
}

void Vibe::apply1(const Img &_image,
                  std::vector<std::unique_ptr<Img>> &_bgImg,
                  Img &_fgmask,
                  const VibeParams &_params,
                  Pcg32 &_rndGen)
{
    _fgmask.clear();

    size_t pixOffset{0};
    for (int y{0}; y < _image.size.height; ++y)
    {
        for (int x{0}; x < _image.size.width; ++x, ++pixOffset)
        {
            uint32_t nGoodSamplesCount{0},
                nSampleIdx{0};

            const uint8_t pixData{_image.data[pixOffset]};

            while (nSampleIdx < _params.NBGSamples)
            {
                if (std::abs((int32_t)_bgImg[nSampleIdx]->data[pixOffset] - (int32_t)pixData) < _params.NColorDistThreshold)
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
                    _bgImg[_rndGen.fast() & _params.ANDlearningRate]->data[pixOffset] = pixData;
                }
                if ((_rndGen.fast() & _params.ANDlearningRate) == 0)
                {
                    const int neighData{getNeighborPosition_3x3(x, y, _image.size, _rndGen.fast())};
                    _bgImg[_rndGen.fast() & _params.ANDlearningRate]->data[neighData] = pixData;
                }
            }
        }
    }
}

void Vibe::getBackgroundImage(cv::Mat &backgroundImage)
{
    cv::Mat oAvgBGImg(m_origImgSize->height, m_origImgSize->width, CV_32FC(m_origImgSize->numBytesPerPixel));

    for (size_t t{0}; t < m_numProcessesParallel; ++t)
    {
        const std::vector<std::unique_ptr<Img>> &bgSamples = m_bgImgSamples[t];
        for (size_t n{0}; n < m_params.NBGSamples; ++n)
        {
            size_t inPixOffset{0};
            size_t outPixOffset{bgSamples[0]->size.originalPixelPos * sizeof(float) * bgSamples[0]->size.numBytesPerPixel};
            for (; inPixOffset < bgSamples[n]->size.size;
                 inPixOffset += m_origImgSize->numBytesPerPixel,
                 outPixOffset += sizeof(float) * bgSamples[0]->size.numBytesPerPixel)
            {
                const uint8_t *const pixData{&bgSamples[n]->data[inPixOffset]};
                float *const outData{(float *)(oAvgBGImg.data + outPixOffset)};
                for (int c{0}; c < m_origImgSize->numBytesPerPixel; ++c)
                {
                    outData[c] += (float)pixData[c] / (float)m_params.NBGSamples;
                }
            }
        }
    }

    oAvgBGImg.convertTo(backgroundImage, CV_8U);
}
