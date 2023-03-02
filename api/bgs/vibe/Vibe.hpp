#pragma once

#include "CoreBgs.hpp"
#include "CoreParameters.hpp"
#include "VibeUtils.hpp"
#include "pcg32.hpp"

namespace sky360lib::bgs
{
    class Vibe
        : public CoreBgs
    {
    public:
        Vibe(VibeParams _params = VibeParams(),
             size_t _numProcessesParallel = DETECT_NUMBER_OF_THREADS);

        virtual VibeParams &getParameters() { return m_params; }

        virtual void getBackgroundImage(cv::Mat &_bgImage);

    private:
        virtual void initialize(const cv::Mat &oInitImg);
        virtual void process(const cv::Mat &_image, cv::Mat &_fgmask, int _numProcess);

        VibeParams m_params;

        std::unique_ptr<ImgSize> m_origImgSize;
        std::vector<std::vector<std::unique_ptr<Img>>> m_bgImgSamples;
        std::vector<Pcg32> m_randomGenerators;

        template<class T>
        void initialize(const Img &_initImg, std::vector<std::unique_ptr<Img>> &_bgImgSamples, Pcg32 &_rndGen);
        template<class T>
        static void apply1(const Img &_image, std::vector<std::unique_ptr<Img>> &_bgImgSamples, Img &_fgmask, const VibeParams &_params, Pcg32 &_rndGen);
        template<class T>
        static void apply3(const Img &_image, std::vector<std::unique_ptr<Img>> &_bgImgSamples, Img &_fgmask, const VibeParams &_params, Pcg32 &_rndGen);
    };
}