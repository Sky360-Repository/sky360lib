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
        ImgSize(const ImgSize &_imgSize)
            : ImgSize(_imgSize.width, _imgSize.height, _imgSize.numBytesPerPixel, _imgSize.originalPixelPos)
        {
        }

        ImgSize(int _width, int _height, int _numBytesPerPixel, size_t _originalPixelPos = 0)
            : width(_width),
              height(_height),
              numBytesPerPixel(_numBytesPerPixel),
              numPixels(_width * _height),
              size(_width * _height * _numBytesPerPixel),
              originalPixelPos{_originalPixelPos}
        {
        }

        static std::unique_ptr<ImgSize> create(int _width, int _height, int _numBytesPerPixel, size_t _originalPixelPos = 0)
        {
            return std::make_unique<ImgSize>(_width, _height, _numBytesPerPixel, _originalPixelPos);
        }

        const int width;
        const int height;
        const int numBytesPerPixel;
        const size_t numPixels;
        const size_t size;

        const size_t originalPixelPos;
    };

    struct Img
    {
        Img(uint8_t *_data, const ImgSize &_imgSize, std::unique_ptr<uint8_t[]> _dataPtr = nullptr)
            : data{_data},
              size{_imgSize},
              dataPtr{std::move(_dataPtr)}
        {
        }

        static std::unique_ptr<Img> create(const ImgSize &_imgSize, bool _clear = false)
        {
            auto data = std::make_unique_for_overwrite<uint8_t[]>(_imgSize.size);
            if (_clear)
            {
                memset(data.get(), 0, _imgSize.size);
            }

            return std::make_unique<Img>(data.get(), _imgSize, std::move(data));
        }

        inline void clear()
        {
            memset(data, 0, size.size);
        }

        uint8_t *const data;
        const ImgSize size;

        std::unique_ptr<uint8_t[]> dataPtr;
    };

    // Returning the number of available threads for the CPU
    inline size_t calcAvailableThreads()
    {
        return (size_t)std::max(1U, std::thread::hardware_concurrency());
    }

    inline bool rectsOverlap(const cv::Rect &r1, const cv::Rect &r2)
    {
        // checking if they don't everlap
        if ((r1.width == 0 || r1.height == 0 || r2.width == 0 || r2.height == 0) ||
            (r1.x > (r2.x + r2.width) || r2.x > (r1.x + r1.width)) ||
            (r1.y > (r2.y + r2.height) || r2.y > (r1.y + r1.height)))
            return false;

        return true;
    }

    inline float rectsDistanceSquared(const cv::Rect &r1, const cv::Rect &r2)
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