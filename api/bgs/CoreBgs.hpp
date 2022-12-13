#pragma once

#include "bgsUtils.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace sky360lib::bgs {

    class CoreBgs {
    public:
        CoreBgs(size_t _numProcessesParallel = 1);

        void apply(const cv::Mat& _image, cv::Mat& _fgmask);

        virtual void getBackgroundImage(cv::Mat& _bgImage) = 0;

    protected:
        virtual void initialize(const cv::Mat& _image) = 0;
        virtual void process(const cv::Mat& _image, cv::Mat& _fgmask, int _numProcess) = 0;

        void prepareParallel(const cv::Mat& _image);
        void applyParallel(const cv::Mat& _image, cv::Mat& _fgmask);

        size_t m_numProcessesParallel;
        bool m_initialized;
        std::vector<size_t> m_processSeq;
        std::vector<std::unique_ptr<ImgSize>> m_imgSizesParallel;
    };

}