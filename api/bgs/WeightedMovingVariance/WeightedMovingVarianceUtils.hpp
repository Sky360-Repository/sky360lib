#pragma once

#include "CoreParameters.hpp"

namespace sky360lib::bgs
{
    class WMVParams : public CoreParameters
    {
    public:
        enum ParamType
        {
            ThresholdType,
            WeightsType,
            EnableWeightType,
            EnableThresholdType
        };
        static inline const bool DEFAULT_ENABLE_WEIGHT{true};
        static inline const bool DEFAULT_ENABLE_THRESHOLD{true};
        static inline const float DEFAULT_THRESHOLD_VALUE{30.0f};
        static inline const float DEFAULT_WEIGHTS[] = {0.5f, 0.3f, 0.2f};
        static inline const float ONE_THIRD{1.0f / 3.0f};

        WMVParams()
            : WMVParams(DEFAULT_ENABLE_WEIGHT, 
                DEFAULT_ENABLE_THRESHOLD, 
                DEFAULT_THRESHOLD_VALUE,
                DEFAULT_ENABLE_WEIGHT ? DEFAULT_WEIGHTS[0] : ONE_THIRD, 
                DEFAULT_ENABLE_WEIGHT ? DEFAULT_WEIGHTS[1] : ONE_THIRD, 
                DEFAULT_ENABLE_WEIGHT ? DEFAULT_WEIGHTS[2] : ONE_THIRD)
        {
        }

        WMVParams(bool _enableWeight,
                bool _enableThreshold,
                float _threshold,
                float _weight1,
                float _weight2,
                float _weight3)
            : CoreParameters()
            , enableWeight{_enableWeight}
            , enableThreshold{_enableThreshold}
            , threshold{_threshold}
            , threshold16{_threshold * 256.0f}
            , weight{_enableWeight ? _weight1 : ONE_THIRD, 
                _enableWeight ? _weight2 : ONE_THIRD, 
                _enableWeight ? _weight3 : ONE_THIRD, 
                0.0f}
            , thresholdSquared{_threshold * _threshold}
            , thresholdSquared16{(_threshold * 256.0f) * (_threshold * 256.0f)}
        {
            setMap();
        }

        WMVParams(const WMVParams& _params)
            : CoreParameters()
            , enableWeight{_params.enableWeight}
            , enableThreshold{_params.enableThreshold}
            , threshold{_params.threshold}
            , threshold16{_params.threshold16}
            , weight{_params.weight[0], _params.weight[1], _params.weight[2], _params.weight[3]}
            , thresholdSquared{_params.thresholdSquared}
            , thresholdSquared16{_params.thresholdSquared16}
        {
            setMap();
        }        

        friend class WeightedMovingVariance;
        friend class WeightedMovingVarianceCL;        

    protected:
        bool enableWeight;
        bool enableThreshold;
        float threshold;
        float threshold16;
        float weight[4];
        float thresholdSquared;
        float thresholdSquared16;

        void setMap()
        {
            m_paramsMap.insert({ParamType::EnableWeightType, ParamMap("EnableWeight", false, &enableWeight)});
            m_paramsMap.insert({ParamType::EnableThresholdType, ParamMap("EnableThreshold", false, &enableThreshold)});
            m_paramsMap.insert({ParamType::ThresholdType, ParamMap("Threshold", false, &threshold)});
            m_paramsMap.insert({ParamType::WeightsType, ParamMap("Weights", false, &weight)});
        }

        virtual void paramUpdated(int _param)
        {
            if (_param == ParamType::ThresholdType)
            {
                threshold16 = threshold * 256.0f;
                thresholdSquared = threshold * threshold;
                thresholdSquared16 = threshold16 * threshold16;
            }
        }
    };
}
