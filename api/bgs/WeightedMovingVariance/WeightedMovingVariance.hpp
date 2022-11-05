#pragma once

#include "CoreBgs.hpp"
#include "WeightedMovingVarianceUtils.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs
{
    class WeightedMovingVariance
        : public CoreBgs
    {
    public:
        static inline const bool DEFAULT_ENABLE_WEIGHT{true};
        static inline const bool DEFAULT_ENABLE_THRESHOLD{true};
        static inline const float DEFAULT_THRESHOLD_VALUE{15.0f};
        /// defines the default value for the number of parallel threads
        static inline const size_t DEFAULT_PARALLEL_TASKS{12};
        static inline const float DEFAULT_WEIGHTS[] = {0.5f, 0.3f, 0.2f};

        WeightedMovingVariance(bool _enableWeight = DEFAULT_ENABLE_WEIGHT,
                               bool _enableThreshold = DEFAULT_ENABLE_THRESHOLD,
                               float _threshold = DEFAULT_THRESHOLD_VALUE,
                               size_t _numProcessesParallel = DEFAULT_PARALLEL_TASKS);
        ~WeightedMovingVariance();

        void getBackgroundImage(cv::Mat &_bgImage);

    private:
        void initialize(const cv::Mat &_image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);

        static inline const float ONE_THIRD{1.0f / 3.0f};

        std::vector<std::array<std::unique_ptr<cv::Mat>, 2>> imgInputPrevParallel;
        const WeightedMovingVarianceParams m_params;

        static void process(const cv::Mat &img_input,
                            cv::Mat &img_output,
                            std::array<std::unique_ptr<cv::Mat>, 2> &img_input_prev,
                            const WeightedMovingVarianceParams &_params);
        static void weightedVarianceMono(
            const uint8_t *const img1,
            const uint8_t *const img2,
            const uint8_t *const img3,
            uint8_t *const outImg,
            const size_t totalPixels,
            const WeightedMovingVarianceParams &_params);
        static void weightedVarianceColor(
            const uint8_t *const img1,
            const uint8_t *const img2,
            const uint8_t *const img3,
            uint8_t *const outImg,
            const size_t totalPixels,
            const WeightedMovingVarianceParams &_params);
    };
}
