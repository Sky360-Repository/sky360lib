#pragma once

#include "coreUtils.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace sky360lib::bgs
{
    class CoreBgs
    {
    public:
        /// Detects the number of available threads to use
        /// Will set the number fo threads to the number of avaible threads - 1
        static const size_t DETECT_NUMBER_OF_THREADS{0};

        CoreBgs(size_t _numProcessesParallel = DETECT_NUMBER_OF_THREADS);

        void apply(const cv::Mat &_image, cv::Mat &_fgmask);
        cv::Mat applyRet(const cv::Mat &_image);

        virtual void getBackgroundImage(cv::Mat &_bgImage) = 0;

    protected:
        virtual void initialize(const cv::Mat &_image) = 0;
        virtual void process(const cv::Mat &_image, cv::Mat &_fgmask, int _numProcess) = 0;

        void prepareParallel(const cv::Mat &_image);
        void applyParallel(const cv::Mat &_image, cv::Mat &_fgmask);

        size_t m_numProcessesParallel;
        bool m_initialized;
        std::vector<size_t> m_processSeq;
        std::vector<std::unique_ptr<ImgSize>> m_imgSizesParallel;
    };

}