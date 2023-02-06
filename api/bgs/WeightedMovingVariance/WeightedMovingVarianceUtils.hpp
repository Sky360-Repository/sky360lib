#pragma once

struct WeightedMovingVarianceParams
{
    static inline const bool DEFAULT_ENABLE_WEIGHT{true};
    static inline const bool DEFAULT_ENABLE_THRESHOLD{true};
    static inline const float DEFAULT_THRESHOLD_VALUE{30.0f};
    static inline const float DEFAULT_WEIGHTS[] = {0.5f, 0.3f, 0.2f};
    static inline const float ONE_THIRD{1.0f / 3.0f};

    WeightedMovingVarianceParams()
        : WeightedMovingVarianceParams(DEFAULT_ENABLE_WEIGHT, 
            DEFAULT_ENABLE_THRESHOLD, 
            DEFAULT_THRESHOLD_VALUE,
            DEFAULT_ENABLE_WEIGHT ? DEFAULT_WEIGHTS[0] : ONE_THIRD, 
            DEFAULT_ENABLE_WEIGHT ? DEFAULT_WEIGHTS[1] : ONE_THIRD, 
            DEFAULT_ENABLE_WEIGHT ? DEFAULT_WEIGHTS[2] : ONE_THIRD)
    {
    }

    WeightedMovingVarianceParams(bool _enableWeight,
                                bool _enableThreshold,
                                float _threshold,
                                float _weight1,
                                float _weight2,
                                float _weight3)
        : enableWeight{_enableWeight},
        enableThreshold{_enableThreshold},
        threshold{_threshold},
        threshold16{_threshold * 256.0f},
        weight{_enableWeight ? _weight1 : ONE_THIRD, 
            _enableWeight ? _weight2 : ONE_THIRD, 
            _enableWeight ? _weight3 : ONE_THIRD, 
            0.0f},
        thresholdSquared{_threshold * _threshold},
        thresholdSquared16{(_threshold * 256.0f) * (_threshold * 256.0f)}
    {
    }

    const bool enableWeight;
    const bool enableThreshold;
    const float threshold;
    const float threshold16;
    const float weight[4];
    const float thresholdSquared;
    const float thresholdSquared16;
};
