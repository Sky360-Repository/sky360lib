#pragma once

struct WeightedMovingVarianceParams
{
    WeightedMovingVarianceParams(bool _enableWeight,
                                    bool _enableThreshold,
                                    float _threshold,
                                    float _weight1,
                                    float _weight2,
                                    float _weight3)
        : enableWeight{_enableWeight},
        enableThreshold{_enableThreshold},
        threshold{_threshold},
        weight{_weight1, _weight2, _weight3, 0.0f},
        thresholdSquared{_threshold * _threshold}
    {
    }

    const bool enableWeight;
    const bool enableThreshold;
    const float threshold;
    const float weight[4];
    const float thresholdSquared;
};
