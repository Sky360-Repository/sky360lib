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

        uint8_t* m_pImgInputCuda;
        uint8_t* m_pImgInputPrev1Cuda;
        uint8_t* m_pImgInputPrev2Cuda;

        size_t m_currentRollingIdx;
        uint8_t* m_pImgOutputCuda;
        uint8_t* m_pImgMemCuda[3];

        int m_firstPhase;

        static inline const float ONE_THIRD{1.0f / 3.0f};

        static void weightedVarianceMono(
            const uint8_t* const _img1,
            const uint8_t* const _img2,
            const uint8_t* const _img3,
            uint8_t* const _outImg,
            const size_t numPixels,
            const WeightedMovingVarianceParams &_params);
        static void weightedVarianceColor(
            const uint8_t* const _img1,
            const uint8_t* const _img2,
            const uint8_t* const _img3,
            uint8_t* const _outImg,
            const size_t numPixels,
            const WeightedMovingVarianceParams &_params);
    };
}
