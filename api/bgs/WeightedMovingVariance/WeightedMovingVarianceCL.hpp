#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 210
#include "CoreBgs.hpp"
#include "WeightedMovingVarianceUtils.hpp"

#include <opencv2/opencv.hpp>
#include <CL/opencl.hpp>

#include <array>
#include <vector>

namespace sky360lib::bgs
{
    class WeightedMovingVarianceCL final
        : public CoreBgs
    {
    public:
        WeightedMovingVarianceCL(const WeightedMovingVarianceParams& _params = WeightedMovingVarianceParams());
        ~WeightedMovingVarianceCL();

        void getBackgroundImage(cv::Mat &_bgImage);

    private:
        void initialize(const cv::Mat &_image);
        void process(const cv::Mat &img_input, cv::Mat &img_output, int _numProcess);
        void initOpenCL();
        void clearCL();

        static const inline int ROLLING_BG_IDX[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};

        const WeightedMovingVarianceParams m_params;

        struct RollingImages
        {
            size_t currentRollingIdx;
            int firstPhase;
            ImgSize* pImgSize;
            cl::Buffer* pImgInput;
            cl::Buffer* pImgInputPrev1;
            cl::Buffer* pImgInputPrev2;
            cl::Buffer bImgOutput;
            cl::Buffer bWeight;
            std::array<cl::Buffer, 3> pImgMem;
        };
        std::vector<RollingImages> imgInputPrev;

        cl::Device m_device;
        cl::Context m_context;
        cl::CommandQueue m_queue;
        cl::Program m_program;
        cl::Kernel m_wmvKernel;

        static void rollImages(RollingImages& rollingImages);
        void process(const cv::Mat &_imgInput,
                    cv::Mat &_imgOutput,
                    RollingImages &_imgInputPrev);
    };
}
