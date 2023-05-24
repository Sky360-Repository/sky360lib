#pragma once

#include <opencv2/opencv.hpp>

namespace sky360lib::utils 
{

    struct WhiteBalanceValues 
    {
        double red;
        double green;
        double blue;
    };

    class AutoWhiteBalance 
    {
    public:
        static WhiteBalanceValues grayWorld(const cv::Mat& image, const WhiteBalanceValues& currentWB, double adjustmentFactor, double threshold) 
        {
            const cv::Scalar imgMean = cv::mean(image);

            const double avg = (imgMean[0] + imgMean[1] + imgMean[2]) / 3.0;

            WhiteBalanceValues wbValues = 
            { 
                std::max(0.0, std::min(255.0, currentWB.red * (1.0 + adjustmentFactor * (avg / imgMean[2] - 1.0)))),
                std::max(0.0, std::min(255.0, currentWB.green * (1.0 + adjustmentFactor * (avg / imgMean[1] - 1.0)))),
                std::max(0.0, std::min(255.0, currentWB.blue * (1.0 + adjustmentFactor * (avg / imgMean[0] - 1.0))))
            };

            // Check if the changes are substantial enough
            if (abs(wbValues.red - currentWB.red) > threshold ||
                abs(wbValues.green - currentWB.green) > threshold ||
                abs(wbValues.blue - currentWB.blue) > threshold) 
            {
                return wbValues;
            } 

            return currentWB;
        }
    };
} 
