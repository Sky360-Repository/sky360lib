#pragma once

#include "core.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <cstring>
#include <thread>

namespace sky360lib
{
    struct ImgSize
    {
        ImgSize(const ImgSize& _imgSize)
            : ImgSize(_imgSize.width, _imgSize.height, _imgSize.numChannels, _imgSize.bytesPerPixel, _imgSize.originalPixelPos)
        {
        }

        ImgSize(int _width, int _height, int _numChannels, int _bytesPerPixel, size_t _originalPixelPos)
            : width(_width)
            , height(_height)
            , numChannels(_numChannels)
            , bytesPerPixel(_bytesPerPixel)
            , numPixels(_width * _height)
            , sizeInBytes(_width * _height * _numChannels * _bytesPerPixel)
            , originalPixelPos{_originalPixelPos}
        {
        }

        static std::unique_ptr<ImgSize> create(int _width, int _height, int _numChannels, int _bytesPerPixel, size_t _originalPixelPos)
        {
            return std::make_unique<ImgSize>(_width, _height, _numChannels, _bytesPerPixel, _originalPixelPos);
        }

        const int width;
        const int height;
        const int numChannels;
        const int bytesPerPixel;
        const size_t numPixels;
        const size_t sizeInBytes;

        const size_t originalPixelPos;
    };

    struct Img
    {
        Img(uint8_t* _data, const ImgSize& _imgSize, std::unique_ptr<uint8_t[]> _dataPtr = nullptr)
            : data{_data}
            , size{_imgSize}
            , dataPtr{std::move(_dataPtr)}
        {
        }

        static std::unique_ptr<Img> create(const ImgSize& _imgSize, bool _clear = false)
        {
            auto data = std::make_unique_for_overwrite<uint8_t[]>(_imgSize.sizeInBytes);
            if (_clear)
            {
                memset(data.get(), 0, _imgSize.sizeInBytes);
            }

            return std::make_unique<Img>(data.get(), _imgSize, std::move(data));
        }

        inline void clear()
        {
            memset(data, 0, size.sizeInBytes);
        }

        uint8_t* const data;
        const ImgSize size;

        template<class T>
        inline T* ptr() { return (T*)data; }
        template<class T>
        inline const T* ptr() const { return (T*)data; }

        std::unique_ptr<uint8_t[]> dataPtr;
    };

    // Returning the number of available threads for the CPU
    inline size_t calcAvailableThreads()
    {
        return (size_t)std::max(1U, std::thread::hardware_concurrency());
    }

    inline bool rectsOverlap(const cv::Rect& r1, const cv::Rect& r2)
    {
        // checking if they don't everlap
        if ((r1.width == 0 || r1.height == 0 || r2.width == 0 || r2.height == 0) ||
            (r1.x > (r2.x + r2.width) || r2.x > (r1.x + r1.width)) ||
            (r1.y > (r2.y + r2.height) || r2.y > (r1.y + r1.height)))
            return false;

        return true;
    }

    inline float rectsDistanceSquared(const cv::Rect& r1, const cv::Rect& r2)
    {
        if (rectsOverlap(r1, r2))
            return 0;

        // compute distance on x axis
        const int xDistance = std::max(0, std::max(r1.x, r2.x) - std::min(r1.x + r1.width, r2.x + r2.width));
        // compute distance on y axis
        const int yDistance = std::max(0, std::max(r1.y, r2.y) - std::min(r1.y + r1.height, r2.y + r2.height));

        return (xDistance * xDistance) + (yDistance * yDistance);
    }
}