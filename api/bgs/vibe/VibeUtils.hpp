#pragma once

#include <iostream>
#include <array>
#include <memory>
#include <vector>

namespace sky360lib::bgs
{
    template<class T>
    static inline int64_t L2dist3Squared(const T *const a, const T *const b)
    {
        const int64_t r0{(int64_t)a[0] - (int64_t)b[0]};
        const int64_t r1{(int64_t)a[1] - (int64_t)b[1]};
        const int64_t r2{(int64_t)a[2] - (int64_t)b[2]};
        return (r0 * r0) + (r1 * r1) + (r2 * r2);
    }

    /// returns the neighbor location for the specified random index & original pixel location; also guards against out-of-bounds values via image/border size check
    static inline int getNeighborPosition_3x3_pos(const int pix, const ImgSize &oImageSize, const uint32_t nRandIdx)
    {
        typedef std::array<int, 2> Nb;
        static const std::array<Nb, 8> s_anNeighborPattern = {
            Nb{-1, 1},
            Nb{0, 1},
            Nb{1, 1},
            Nb{-1, 0},
            Nb{1, 0},
            Nb{-1, -1},
            Nb{0, -1},
            Nb{1, -1},
        };
        const size_t r{nRandIdx & 0x7};
        int nNeighborCoord_X{std::max(std::min((pix % oImageSize.width) + s_anNeighborPattern[r][0], oImageSize.width - 1), 0)};
        int nNeighborCoord_Y{std::max(std::min((pix / oImageSize.width) + s_anNeighborPattern[r][1], oImageSize.height - 1), 0)};
        return (nNeighborCoord_Y * oImageSize.width + nNeighborCoord_X);
    }

    static inline int getNeighborPosition_3x3(const int x, const int y, const ImgSize &oImageSize, const uint32_t nRandIdx)
    {
        typedef std::array<int, 2> Nb;
        static const std::array<Nb, 8> s_anNeighborPattern = {
            Nb{-1, 1},
            Nb{0, 1},
            Nb{1, 1},
            Nb{-1, 0},
            Nb{1, 0},
            Nb{-1, -1},
            Nb{0, -1},
            Nb{1, -1},
        };
        const size_t r{nRandIdx & 0x7};
        const int nNeighborCoord_X{std::max(std::min(x + s_anNeighborPattern[r][0], oImageSize.width - 1), 0)};
        const int nNeighborCoord_Y{std::max(std::min(y + s_anNeighborPattern[r][1], oImageSize.height - 1), 0)};
        return (nNeighborCoord_Y * oImageSize.width + nNeighborCoord_X);
    }

    /// returns pixel coordinates clamped to the given image & border size
    inline void clampImageCoords(int &nSampleCoord_X, int &nSampleCoord_Y, const ImgSize &oImageSize)
    {
        if (nSampleCoord_X < 0)
            nSampleCoord_X = 0;
        else if (nSampleCoord_X >= oImageSize.width)
            nSampleCoord_X = oImageSize.width - 1;
        if (nSampleCoord_Y < 0)
            nSampleCoord_Y = 0;
        else if (nSampleCoord_Y >= oImageSize.height)
            nSampleCoord_Y = oImageSize.height - 1;
    }

    /// returns the sampling location for the specified random index & original pixel location, given a predefined kernel; also guards against out-of-bounds values via image/border size check
    template <int nKernelHeight, int nKernelWidth>
    inline void getSamplePosition(const std::array<std::array<int, nKernelWidth>, nKernelHeight> &anSamplesInitPattern,
                                  const int nSamplesInitPatternTot, const int nRandIdx, int &nSampleCoord_X, int &nSampleCoord_Y,
                                  const int nOrigCoord_X, const int nOrigCoord_Y, const ImgSize &oImageSize)
    {
        int r = 1 + (nRandIdx % nSamplesInitPatternTot);
        for (nSampleCoord_Y = 0; nSampleCoord_Y < nKernelHeight; ++nSampleCoord_Y)
        {
            for (nSampleCoord_X = 0; nSampleCoord_X < nKernelWidth; ++nSampleCoord_X)
            {
                r -= anSamplesInitPattern[nSampleCoord_Y][nSampleCoord_X];
                if (r <= 0)
                    goto stop;
            }
        }
    stop:
        nSampleCoord_X += nOrigCoord_X - nKernelWidth / 2;
        nSampleCoord_Y += nOrigCoord_Y - nKernelHeight / 2;
        clampImageCoords(nSampleCoord_X, nSampleCoord_Y, oImageSize);
    }

    /// returns the sampling location for the specified random index & original pixel location; also guards against out-of-bounds values via image/border size check
    inline void getSamplePosition_7x7_std2(const int nRandIdx,
                                           int &nSampleCoord_X, int &nSampleCoord_Y,
                                           const int nOrigCoord_X, const int nOrigCoord_Y,
                                           const ImgSize &oImageSize)
    {
        // based on 'floor(fspecial('gaussian',7,2)*512)'
        static const int s_nSamplesInitPatternTot = 512;
        static const std::array<std::array<int, 7>, 7> s_anSamplesInitPattern = {
            std::array<int, 7>{ 2,  4,  6,  7,  6,  4,  2 },
            std::array<int, 7>{ 4,  8, 12, 14, 12,  8,  4 },
            std::array<int, 7>{ 6, 12, 21, 25, 21, 12,  6 },
            std::array<int, 7>{ 7, 14, 25, 28, 25, 14,  7 },
            std::array<int, 7>{ 6, 12, 21, 25, 21, 12,  6 },
            std::array<int, 7>{ 4,  8, 12, 14, 12,  8,  4 },
            std::array<int, 7>{ 2,  4,  6,  7,  6,  4,  2 },
        };
        getSamplePosition<7, 7>(s_anSamplesInitPattern, s_nSamplesInitPatternTot, nRandIdx, nSampleCoord_X, nSampleCoord_Y, nOrigCoord_X, nOrigCoord_Y, oImageSize);
    }

    static inline void splitImg(const Img &_inputImg, std::vector<std::unique_ptr<Img>> &_outputImages, int _numSplits)
    {
        _outputImages.resize(_numSplits);
        int y = 0;
        int h = _inputImg.size.height / _numSplits;
        for (int i = 0; i < _numSplits; ++i)
        {
            if (i == (_numSplits - 1))
            {
                h = _inputImg.size.height - y;
            }
            _outputImages[i] = Img::create(ImgSize(_inputImg.size.width, h, _inputImg.size.numChannels, _inputImg.size.bytesPerPixel, y * _inputImg.size.width), false);

            memcpy(_outputImages[i]->data,
                   _inputImg.data + (_outputImages[i]->size.originalPixelPos * _inputImg.size.numChannels * _inputImg.size.bytesPerPixel),
                   _outputImages[i]->size.sizeInBytes);
            y += h;
        }
    }

    struct VibeParams2
    {
        /// defines the default value for ColorDistThreshold
        static const size_t DEFAULT_COLOR_DIST_THRESHOLD{50}; //{32};
        /// defines the default value for BGSamples
        static const size_t DEFAULT_NB_BG_SAMPLES{16}; //{8};
        /// defines the default value for RequiredBGSamples
        static const size_t DEFAULT_REQUIRED_NB_BG_SAMPLES{1}; //{2};
        /// defines the default value for the learning rate passed to the 'subsampling' factor in the original ViBe paper
        static const size_t DEFAULT_LEARNING_RATE{2}; //{8};

        VibeParams2()
            : VibeParams2(DEFAULT_COLOR_DIST_THRESHOLD, DEFAULT_NB_BG_SAMPLES, DEFAULT_REQUIRED_NB_BG_SAMPLES, DEFAULT_LEARNING_RATE)
        {
        }

        VibeParams2(uint32_t nColorDistThreshold,
                   uint32_t nBGSamples,
                   uint32_t nRequiredBGSamples,
                   uint32_t learningRate)
            : NBGSamples(nBGSamples),
              NRequiredBGSamples(nRequiredBGSamples),
              NColorDistThresholdMono(nColorDistThreshold),
              NColorDistThresholdColorSquared{(nColorDistThreshold * 3) * (nColorDistThreshold * 3)},
              NColorDistThresholdMono16(nColorDistThreshold << 8),
              NColorDistThresholdColor16Squared{((nColorDistThreshold << 8) * 3) * ((nColorDistThreshold << 8) * 3)},
              LearningRate(learningRate),
              ANDlearningRate{learningRate - 1}
        {
        }

        /// number of different samples per pixel/block to be taken from input frames to build the background model ('N' in the original ViBe paper)
        const uint32_t NBGSamples;
        /// number of similar samples needed to consider the current pixel/block as 'background' ('#_min' in the original ViBe paper)
        const uint32_t NRequiredBGSamples;
        /// absolute color distance threshold ('R' or 'radius' in the original ViBe paper)
        const int32_t NColorDistThresholdMono;
        const int64_t NColorDistThresholdColorSquared;
        const int32_t NColorDistThresholdMono16;
        const int64_t NColorDistThresholdColor16Squared;
        /// should be > 0 and factor of 2 (smaller values == faster adaptation)
        const uint32_t LearningRate;
        const uint32_t ANDlearningRate;
    };

    class VibeParams : public CoreParameters
    {
    public:
        enum ParamType
        {
            ColorDistThresholdType,
            BGSamplesType,
            RequiredBGSamplesType,
            LearningRateType
        };
        /// defines the default value for ColorDistThreshold
        static const uint32_t DEFAULT_COLOR_DIST_THRESHOLD{50}; //{32};
        /// defines the default value for BGSamples
        static const uint32_t DEFAULT_NB_BG_SAMPLES{16}; //{8};
        /// defines the default value for RequiredBGSamples
        static const uint32_t DEFAULT_REQUIRED_NB_BG_SAMPLES{1}; //{2};
        /// defines the default value for the learning rate passed to the 'subsampling' factor in the original ViBe paper
        static const uint32_t DEFAULT_LEARNING_RATE{2}; //{8};

        VibeParams()
            : VibeParams(DEFAULT_COLOR_DIST_THRESHOLD, 
                DEFAULT_NB_BG_SAMPLES, 
                DEFAULT_REQUIRED_NB_BG_SAMPLES, 
                DEFAULT_LEARNING_RATE)
        {
        }

        VibeParams(uint32_t nColorDistThreshold,
                   uint32_t nBGSamples,
                   uint32_t nRequiredBGSamples,
                   uint32_t learningRate)
            : CoreParameters()
            , NBGSamples{nBGSamples}
            , NRequiredBGSamples{nRequiredBGSamples}
            , NColorDistThresholdMono{nColorDistThreshold}
            , NColorDistThresholdColorSquared{((uint64_t)nColorDistThreshold * 3) * ((uint64_t)nColorDistThreshold * 3)}
            , NColorDistThresholdMono16{nColorDistThreshold << 8}
            , NColorDistThresholdColor16Squared{(((uint64_t)nColorDistThreshold << 8) * 3) * (((uint64_t)nColorDistThreshold << 8) * 3)}
            , LearningRate{learningRate}
            , ANDlearningRate{learningRate - 1}
        {
            setMap();
        }

        VibeParams(const VibeParams& _params)
            : CoreParameters()
            , NBGSamples{_params.NBGSamples}
            , NRequiredBGSamples{_params.NRequiredBGSamples}
            , NColorDistThresholdMono{_params.NColorDistThresholdMono}
            , NColorDistThresholdColorSquared{_params.NColorDistThresholdColorSquared}
            , NColorDistThresholdMono16{_params.NColorDistThresholdMono16}
            , NColorDistThresholdColor16Squared{_params.NColorDistThresholdColor16Squared}
            , LearningRate{_params.LearningRate}
            , ANDlearningRate{_params.ANDlearningRate}
        {
            setMap();
        }

        friend class Vibe;

    protected:
        /// number of different samples per pixel/block to be taken from input frames to build the background model ('N' in the original ViBe paper)
        uint32_t NBGSamples;
        /// number of similar samples needed to consider the current pixel/block as 'background' ('#_min' in the original ViBe paper)
        uint32_t NRequiredBGSamples;
        /// absolute color distance threshold ('R' or 'radius' in the original ViBe paper)
        uint32_t NColorDistThresholdMono;
        uint64_t NColorDistThresholdColorSquared;
        uint32_t NColorDistThresholdMono16;
        uint64_t NColorDistThresholdColor16Squared;
        /// should be > 0 and factor of 2 (smaller values == faster adaptation)
        uint32_t LearningRate;
        uint32_t ANDlearningRate;

        void setMap()
        {
            m_paramsMap.clear();
            m_paramsMap.insert({ParamType::ColorDistThresholdType, ParamMap("ColorDistThreshold", false, &NColorDistThresholdMono)});
            m_paramsMap.insert({ParamType::BGSamplesType, ParamMap("BGSamples", true, &NBGSamples)});
            m_paramsMap.insert({ParamType::RequiredBGSamplesType, ParamMap("RequiredBGSamples", true, &NRequiredBGSamples)});
            m_paramsMap.insert({ParamType::LearningRateType, ParamMap("LearningRate", false, &LearningRate)});
        }

        virtual void paramUpdated(int _param)
        {
            switch (_param)
            {
                case ParamType::ColorDistThresholdType:
                    NColorDistThresholdColorSquared = (NColorDistThresholdMono * 3) * (NColorDistThresholdMono * 3);
                    NColorDistThresholdMono16 = NColorDistThresholdMono * 256;
                    NColorDistThresholdColor16Squared = (NColorDistThresholdMono16 * 3) * (NColorDistThresholdMono16 * 3);
                    break;
                case ParamType::LearningRateType:
                    ANDlearningRate = LearningRate - 1;
                    break;
            }
        }
    };
}