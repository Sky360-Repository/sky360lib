#pragma once

#include <opencv2/opencv.hpp>

namespace sky360lib {
namespace utils {

        struct WhiteBalanceValues {
            int red;
            int green;
            int blue;
        };

        class AutoWhiteBalance {
        private:
            const double adjustmentFactor = 1;

        public:
            WhiteBalanceValues grayWorld(cv::Mat& image, WhiteBalanceValues currentWB, double adjustmentFactor, int threshold) 
            {
                cv::Scalar imgMean = cv::mean(image);

                double avg = (imgMean[0] + imgMean[1] + imgMean[2]) / 3.0;

                WhiteBalanceValues wbValues;

                wbValues.red = static_cast<int>(currentWB.red * (1 + adjustmentFactor * (avg / imgMean[2] - 1)));
                wbValues.green = static_cast<int>(currentWB.green * (1 + adjustmentFactor * (avg / imgMean[1] - 1)));
                wbValues.blue = static_cast<int>(currentWB.blue * (1 + adjustmentFactor * (avg / imgMean[0] - 1)));

                // Keep the values within valid range
                wbValues.red = std::max(0, std::min(255, wbValues.red));
                wbValues.green = std::max(0, std::min(255, wbValues.green));
                wbValues.blue = std::max(0, std::min(255, wbValues.blue));

                // Check if the changes are substantial enough
                if (abs(wbValues.red - currentWB.red) > threshold ||
                    abs(wbValues.green - currentWB.green) > threshold ||
                    abs(wbValues.blue - currentWB.blue) > threshold) {
                    return wbValues;
                } else {
                    return currentWB;
                }
            
            }
        };
    } 
} 
