#pragma once

#include "core.hpp"

#include <memory>

namespace sky360lib::bgs {

    struct ImgSize {
        ImgSize(const ImgSize& _imgSize)
            : ImgSize(_imgSize.width, _imgSize.height, _imgSize.numBytesPerPixel, _imgSize.originalPixelPos) {
        }

        ImgSize(int _width, int _height, int _numBytesPerPixel, int _originalPixelPos = 0) 
            : width(_width), 
            height(_height),
            numBytesPerPixel(_numBytesPerPixel),
            numPixels(_width * _height),
            size(_width * _height * _numBytesPerPixel),
            originalPixelPos{_originalPixelPos}
        {}

        static std::unique_ptr<ImgSize> create(int _width, int _height, int _numBytesPerPixel, int _originalPixelPos = 0) {
            return std::make_unique<ImgSize>(_width, _height, _numBytesPerPixel, _originalPixelPos);
        }

        const int width;
        const int height;
        const int numBytesPerPixel;
        const int numPixels;
        const int size;

        const int originalPixelPos;
    };

}