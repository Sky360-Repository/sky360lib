#pragma once

#include "CoreBgs.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs
{
    struct WeightedMovingVarianceParams
    {
        WeightedMovingVarianceParams(bool _enableWeight,
                                     bool _enableThreshold,
                                     float _threshold,
                                     float _weight1,
                                     float _weight2,
                                     float _weight3)
            : enableWeight(_enableWeight),
              enableThreshold(_enableThreshold),
              threshold(_threshold),
              weight1(_weight1),
              weight2(_weight2),
              weight3(_weight3)
        {
        }

        const bool enableWeight;
        const bool enableThreshold;
        const float threshold;

        const float weight1;
        const float weight2;
        const float weight3;
    };

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
            const cv::Mat &_img1,
            const cv::Mat &_img2,
            const cv::Mat &_img3,
            cv::Mat &_outImg,
            const WeightedMovingVarianceParams &_params);
        static void weightedVarianceColor(
            const cv::Mat &_img1,
            const cv::Mat &_img2,
            const cv::Mat &_img3,
            cv::Mat &_outImg,
            const WeightedMovingVarianceParams &_params);
    };
}
