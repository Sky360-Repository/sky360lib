#pragma once

#include "CoreParameters.hpp"

namespace sky360lib::bgs
{
    class WMVParams : public CoreParameters
    {
    public:
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
            , weight{_enableWeight ? _weight1 : ONE_THIRD, 
                _enableWeight ? _weight2 : ONE_THIRD, 
                _enableWeight ? _weight3 : ONE_THIRD}
        {
            setEnableThreshold(_enableThreshold);
            setEnableWeight(_enableWeight);
            setThreshold(_threshold);
        }

        WMVParams(const WMVParams& _params)
            : CoreParameters()
            , weight{_params.weight[0], _params.weight[1], _params.weight[2]}
        {
            setEnableThreshold(_params.enableThreshold);
            setEnableWeight(_params.enableWeight);
            setThreshold(_params.threshold);
        }

        // WMVParams& operator=(const WMVParams& _params)
        // {
        //     enableWeight = _params.enableWeight;
        //     enableThreshold = _params.enableThreshold;
        //     threshold = _params.threshold;
        //     threshold16 = _params.threshold16;
        //     weight[0] = _params.weight[0];
        //     weight[1] = _params.weight[1];
        //     weight[2] = _params.weight[2];
        //     thresholdSquared = _params.thresholdSquared;
        //     thresholdSquared16 = _params.thresholdSquared16;
        //     setMap();
        //     return *this;
        // }

        float getThreshold() { return threshold; }
        float* getWeights() { return weight; }
        bool getEnableWeight() { return enableWeight; }
        bool getEnableThreshold() { return enableThreshold; }

        void setEnableWeight(bool value)
        { 
            enableWeight = value; 
        }
        void setEnableThreshold(bool value)
        { 
            enableThreshold = value; 
        }
        void setWeights(int _weight, float _value)
        {
            if (_weight >= 0 && _weight <= 3)
            {
                weight[_weight] = _value; 
            }
        }
        void setThreshold(float _value) 
        { 
            threshold = _value;
            threshold16 = threshold * 256.0f;
            thresholdSquared = threshold * threshold;
            thresholdSquared16 = threshold16 * threshold16;
        }

        friend class WeightedMovingVariance;
        friend class WeightedMovingVarianceCL;        

    protected:
        bool enableWeight;
        bool enableThreshold;
        float threshold;
        float threshold16;
        float weight[3];
        float thresholdSquared;
        float thresholdSquared16;
    };
}
