#include "CoreBgs.hpp"
#include "CoreParameters.hpp"

#include <iostream>
#include <execution>
#include <algorithm>

using namespace sky360lib::bgs;

CoreBgs::CoreBgs(size_t _numProcessesParallel)
    : m_numProcessesParallel{_numProcessesParallel}
    , m_initialized{false}
{
    if (_numProcessesParallel == DETECT_NUMBER_OF_THREADS)
    {
        m_numProcessesParallel = calcAvailableThreads();;
    }
}

void CoreBgs::restart()
{
    m_initialized = false;
}

void CoreBgs::apply(const cv::Mat &_image, cv::Mat &_fgmask)
{
    if (!m_initialized)
    {
        prepareParallel(_image);
        initialize(_image);
        m_initialized = true;
    }
    if (_fgmask.empty())
    {
        _fgmask.create(_image.size(), CV_8UC1);
    }

    if (m_numProcessesParallel == 1)
    {
        //std::cout << "CoreBgs runing in the same thread" << std::endl;
        process(_image, _fgmask, 0);
    }
    else
    {
        //std::cout << "CoreBgs runing in " << m_numProcessesParallel << " threads" << std::endl;
        applyParallel(_image, _fgmask);
    }
}

cv::Mat CoreBgs::applyRet(const cv::Mat &_image)
{
    cv::Mat imgMask;
    apply(_image, imgMask);
    return imgMask;
}

void CoreBgs::prepareParallel(const cv::Mat &_image)
{
    m_imgSizesParallel.resize(m_numProcessesParallel);
    m_processSeq.resize(m_numProcessesParallel);
    size_t y{0};
    size_t h{_image.size().height / m_numProcessesParallel};
    for (size_t i{0}; i < m_numProcessesParallel; ++i)
    {
        m_processSeq[i] = i;
        if (i == (m_numProcessesParallel - 1))
        {
            h = _image.size().height - y;
        }
        m_imgSizesParallel[i] = ImgSize::create(_image.size().width, h,
                                                _image.channels(),
                                                _image.elemSize1(),
                                                y * _image.size().width);
        y += h;
    }
}

void CoreBgs::applyParallel(const cv::Mat &_image, cv::Mat &_fgmask)
{
    std::for_each(
        std::execution::par,
        m_processSeq.begin(),
        m_processSeq.end(),
        [&](int np)
        {
            const cv::Mat imgSplit(m_imgSizesParallel[np]->height, m_imgSizesParallel[np]->width, _image.type(),
                                   _image.data + (m_imgSizesParallel[np]->originalPixelPos * m_imgSizesParallel[np]->numChannels * m_imgSizesParallel[np]->bytesPerPixel));
            cv::Mat maskPartial(m_imgSizesParallel[np]->height, m_imgSizesParallel[np]->width, _fgmask.type(),
                                _fgmask.data + m_imgSizesParallel[np]->originalPixelPos);
            process(imgSplit, maskPartial, np);
        });
}
