#pragma once

#include "CoreBgs.hpp"
#include "WeightedMovingVarianceUtils.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs
{
    class WeightedMovingVarianceHalide
        : public CoreBgs
    {
    public:
        static inline const bool DEFAULT_ENABLE_WEIGHT{true};
        static inline const bool DEFAULT_ENABLE_THRESHOLD{true};
        static inline const float DEFAULT_THRESHOLD_VALUE{15.0f};
        static inline const float DEFAULT_WEIGHTS[] = {0.5f, 0.3f, 0.2f};

        WeightedMovingVarianceHalide(bool _enableWeight = DEFAULT_ENABLE_WEIGHT,
                               bool _enableThreshold = DEFAULT_ENABLE_THRESHOLD,
                               float _threshold = DEFAULT_THRESHOLD_VALUE);
        ~WeightedMovingVarianceHalide();

        void getBackgroundImage(cv::Mat &_bgImage);

    private:
        void initialize(const cv::Mat &_image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);
        void initParallelData();
        void initHalide();

        static inline const float ONE_THIRD{1.0f / 3.0f};
        static const inline int ROLLING_BG_IDX[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};

        const WeightedMovingVarianceParams m_params;

        struct RollingImages
        {
            size_t currentRollingIdx;
            int firstPhase;
            ImgSize* pImgSize;
            uint8_t* pImgInput;
            uint8_t* pImgInputPrev1;
            uint8_t* pImgInputPrev2;

            std::array<std::unique_ptr<uint8_t[]>, 3> pImgMem;
        };
        std::vector<RollingImages> imgInputPrev;

        static void rollImages(RollingImages& rollingImages);
        static void process(const cv::Mat &_imgInput,
                            cv::Mat &_imgOutput,
                            RollingImages &_imgInputPrev,
                            const WeightedMovingVarianceParams &_params);
        static void weightedVarianceMono(
            uint8_t *const img1,
            uint8_t *const img2,
            uint8_t *const img3,
            uint8_t *const outImg,
            const int width,
            const int height,
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
