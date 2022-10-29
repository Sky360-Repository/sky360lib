#pragma once

#include "VibeBGSUtils.hpp"

#include <opencv2/core.hpp>

namespace sky360 {

    class VibeBGS {
    public:
        /// defines the default value for BackgroundSubtractorViBe::m_nColorDistThreshold
        static const size_t DEFAULT_COLOR_DIST_THRESHOLD{20};
        /// defines the default value for BackgroundSubtractorViBe::m_nBGSamples
        static const size_t DEFAULT_NB_BG_SAMPLES{16};
        /// defines the default value for BackgroundSubtractorViBe::m_nRequiredBGSamples
        static const size_t DEFAULT_REQUIRED_NB_BG_SAMPLES{2};
        /// defines the default value for the learning rate passed to BackgroundSubtractorViBe::apply (the 'subsampling' factor in the original ViBe paper)
        static const size_t DEFAULT_LEARNING_RATE{8};
        /// defines the default value for the number of parallel threads
        static const size_t DEFAULT_PARALLEL_TASKS{4};

        VibeBGS(size_t nColorDistThreshold = DEFAULT_COLOR_DIST_THRESHOLD,
                size_t nBGSamples = DEFAULT_NB_BG_SAMPLES,
                size_t nRequiredBGSamples = DEFAULT_REQUIRED_NB_BG_SAMPLES,
                size_t learningRate = DEFAULT_LEARNING_RATE);

        void initialize(const cv::Mat& oInitImg, int _numProcesses = DEFAULT_PARALLEL_TASKS);

        void apply(const cv::Mat& _image, cv::Mat& _fgmask);

        void getBackgroundImage(cv::Mat& backgroundImage) const;

    private:
        Params m_params;

        int m_numProcessesParallel;
        std::vector<int> m_processSeq;

        std::unique_ptr<ImgSize> m_origImgSize;

        std::vector<std::vector<std::unique_ptr<Img>>> m_bgImgSamples;

        void initialize(const Img& _initImg, std::vector<std::unique_ptr<Img>>& _bgImgSamples);
        void applyParallel(const Img& _image, Img& _fgmask);

        static void apply1(const Img& _image, std::vector<std::unique_ptr<Img>>& _bgImgSamples, Img& _fgmask, const Params& _params);
        static void apply3(const Img& _image, std::vector<std::unique_ptr<Img>>& _bgImgSamples, Img& _fgmask, const Params& _params);
    };
}