#pragma once

#include "CoreBgs.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs {
    
    struct WeightedMovingVarianceParams
    {
        WeightedMovingVarianceParams(bool _enableWeight,
                                    bool _enableThreshold,
                                    int _threshold)
            : enableWeight(_enableWeight),
            enableThreshold(_enableThreshold),
            threshold(_threshold)
        {}

        const bool enableWeight;
        const bool enableThreshold;
        const int threshold;
    };

    class WeightedMovingVariance
        : public CoreBgs {
    public:
        static const bool DEFAULT_ENABLE_WEIGHT{true};
        static const bool DEFAULT_ENABLE_THRESHOLD{true};
        static const int DEFAULT_THRESHOLD_VALUE{15};
        /// defines the default value for the number of parallel threads
        static const size_t DEFAULT_PARALLEL_TASKS{12};

        WeightedMovingVariance(bool _enableWeight = DEFAULT_ENABLE_WEIGHT,
                               bool _enableThreshold = DEFAULT_ENABLE_THRESHOLD,
                               int _threshold = DEFAULT_THRESHOLD_VALUE,
                               size_t _numProcessesParallel = DEFAULT_PARALLEL_TASKS);
        ~WeightedMovingVariance();

        void getBackgroundImage(cv::Mat& _bgImage);

    private:
        void initialize(const cv::Mat& _image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);

        static inline const float ONE_THIRD{1.0f / 3.0f};

        std::vector<std::array<std::unique_ptr<cv::Mat>, 2>> imgInputPrevParallel;
        const WeightedMovingVarianceParams m_params;

        //void processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput);
        static void process(const cv::Mat &img_input, 
                            cv::Mat &img_output, 
                            std::array<std::unique_ptr<cv::Mat>, 2>& img_input_prev, 
                            const WeightedMovingVarianceParams& _params);
        static void weightedVarianceMono(
                const cv::Mat &_img1, 
                const cv::Mat &_img2, 
                const cv::Mat &_img3, 
                cv::Mat& _outImg,
                const WeightedMovingVarianceParams& _params);
        static void weightedVarianceColor(
                const cv::Mat &_img1, 
                const cv::Mat &_img2, 
                const cv::Mat &_img3, 
                cv::Mat& _outImg,
                const WeightedMovingVarianceParams& _params);
    };
}
