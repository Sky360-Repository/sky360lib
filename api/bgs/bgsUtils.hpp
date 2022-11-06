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

    struct Img {
        Img(uint8_t* _data, const ImgSize& _imgSize, std::unique_ptr<uint8_t[]> _dataPtr = nullptr)
            : data{_data},
            size{_imgSize},
            dataPtr{std::move(_dataPtr)} {
        }
        
        static std::unique_ptr<Img> create(const ImgSize& _imgSize, bool _clear = false) {
            auto data = std::make_unique_for_overwrite<uint8_t[]>(_imgSize.size);
            if (_clear) {
                memset(data.get(), 0, _imgSize.size);
            }

            return std::make_unique<Img>(data.get(), _imgSize, std::move(data));
        }

        inline void clear() {
            memset(data, 0, size.size);
        }

        const ImgSize size;
        uint8_t* const data;

        std::unique_ptr<uint8_t[]> dataPtr;
    };
}