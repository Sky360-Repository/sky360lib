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
        WeightedMovingVarianceHalide(const WeightedMovingVarianceParams& _params = WeightedMovingVarianceParams());
        ~WeightedMovingVarianceHalide();

        void getBackgroundImage(cv::Mat &_bgImage);

    private:
        void initialize(const cv::Mat &_image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);
        void initParallelData();

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
            const int width,
            const int height,
            const WeightedMovingVarianceParams &_params);
    };
}
