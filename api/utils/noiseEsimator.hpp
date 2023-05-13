#pragma once

#include <opencv2/opencv.hpp>

// Based on work by Immerkær, see - https://www.sciencedirect.com/science/article/abs/pii/S1077314296900600

namespace sky360lib::utils
{
    class NoiseEstimator {
    public:
        double estimate_noise(cv::Mat image) {

            if (image.channels() == 3) {
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            }

            cv::Mat laplacianMask = (cv::Mat_<double>(3,3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);

            cv::Mat laplacianImage;
            cv::filter2D(image, laplacianImage, -1, laplacianMask, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

            double sigma = cv::sum(cv::abs(laplacianImage))[0];

            int H = image.rows - 2;
            int W = image.cols - 2;
            sigma = sigma * sqrt(0.5 * M_PI) / (6.0 * W * H);

            return sigma;
        }
    };
}