#pragma once

#include "CoreBgs.hpp"
#include "WeightedMovingVarianceUtils.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs
{
    class WeightedMovingVarianceCuda
        : public CoreBgs
    {
    public:
        static inline const bool DEFAULT_ENABLE_WEIGHT{true};
        static inline const bool DEFAULT_ENABLE_THRESHOLD{true};
        static inline const float DEFAULT_THRESHOLD_VALUE{15.0f};
        /// defines the default value for the number of parallel threads
        static inline const float DEFAULT_WEIGHTS[] = {0.5f, 0.3f, 0.2f};

        WeightedMovingVarianceCuda(bool _enableWeight = DEFAULT_ENABLE_WEIGHT,
                               bool _enableThreshold = DEFAULT_ENABLE_THRESHOLD,
                               float _threshold = DEFAULT_THRESHOLD_VALUE);
        ~WeightedMovingVarianceCuda();

        void getBackgroundImage(cv::Mat &_bgImage);

    private:
        void initialize(const cv::Mat &_image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);
        void clearCuda();
        void rollImages();

        const WeightedMovingVarianceParams m_params;

        static const inline int ROLLING_BG_IDX[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};

        uchar* m_pImgInputCuda;
        uchar* m_pImgInputPrev1Cuda;
        uchar* m_pImgInputPrev2Cuda;

        size_t m_currentRollingIdx;
        uchar* m_pImgOutputCuda;
        uchar* m_pImgMemCuda[3];

        int m_firstPhase;

        static inline const float ONE_THIRD{1.0f / 3.0f};

        static void weightedVarianceMono(
            uchar* const _img1,
            uchar* const _img2,
            uchar* const _img3,
            uchar* _outImg,
            const size_t numPixels,
            const WeightedMovingVarianceParams &_params);
        static void weightedVarianceColor(
            uchar* const _img1,
            uchar* const _img2,
            uchar* const _img3,
            uchar* _outImg,
            const size_t numPixels,
            const int numChannels,
            const WeightedMovingVarianceParams &_params);
    };
}
