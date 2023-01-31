#pragma once

#include "CoreBgs.hpp"
#include "WeightedMovingVarianceUtils.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs
{
    class WeightedMovingVariance final
        : public CoreBgs 
    {
    public:
        WeightedMovingVariance(const WeightedMovingVarianceParams& _params = WeightedMovingVarianceParams(),
                               size_t _numProcessesParallel = DETECT_NUMBER_OF_THREADS);
        ~WeightedMovingVariance();

        void getBackgroundImage(cv::Mat &_bgImage);

    private:
        void initialize(const cv::Mat &_image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);

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
        template<class T>
        static void weightedVarianceMono(
            const T *const img1,
            const T *const img2,
            const T *const img3,
            uint8_t *const outImg,
            const size_t totalPixels,
            const WeightedMovingVarianceParams &_params);
        template<class T>
        static void weightedVarianceColor(
            const T *const img1,
            const T *const img2,
            const T *const img3,
            uint8_t *const outImg,
            const size_t totalPixels,
            const WeightedMovingVarianceParams &_params);
    };
}
