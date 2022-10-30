#pragma once

#include "CoreBgs.hpp"
#include "VibeUtils.hpp"

namespace sky360lib::bgs {

    class Vibe
        : public CoreBgs {
    public:
        /// defines the default value for ColorDistThreshold
        static const size_t DEFAULT_COLOR_DIST_THRESHOLD{15};
        /// defines the default value for BGSamples
        static const size_t DEFAULT_NB_BG_SAMPLES{8};
        /// defines the default value for RequiredBGSamples
        static const size_t DEFAULT_REQUIRED_NB_BG_SAMPLES{2};
        /// defines the default value for the learning rate passed to the 'subsampling' factor in the original ViBe paper
        static const size_t DEFAULT_LEARNING_RATE{8};
        /// defines the default value for the number of parallel threads
        static const size_t DEFAULT_PARALLEL_TASKS{12};

        Vibe(size_t nColorDistThreshold = DEFAULT_COLOR_DIST_THRESHOLD,
             size_t nBGSamples = DEFAULT_NB_BG_SAMPLES,
             size_t nRequiredBGSamples = DEFAULT_REQUIRED_NB_BG_SAMPLES,
             size_t learningRate = DEFAULT_LEARNING_RATE,
             size_t _numProcessesParallel = DEFAULT_PARALLEL_TASKS);

        void getBackgroundImage(cv::Mat& _bgImage);

    private:
        void initialize(const cv::Mat& oInitImg);
        void process(const cv::Mat& _image, cv::Mat& _fgmask, int _numProcess);

        VibeParams m_params;
        std::unique_ptr<ImgSize> m_origImgSize;
        std::vector<std::vector<std::unique_ptr<Img>>> m_bgImgSamples;

        void initialize(const Img& _initImg, std::vector<std::unique_ptr<Img>>& _bgImgSamples);

        static void apply1(const Img& _image, std::vector<std::unique_ptr<Img>>& _bgImgSamples, Img& _fgmask, const VibeParams& _params);
        static void apply3(const Img& _image, std::vector<std::unique_ptr<Img>>& _bgImgSamples, Img& _fgmask, const VibeParams& _params);
    };
}