#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace sky360lib::utils
{
    class BrightnessEstimator 
    {
    public:
        BrightnessEstimator(int n, int m)
            : m_n(n), m_m(m) {}

        double estimateCurrentBrightness(cv::Mat& image)
        {
            int rows = image.rows;
            int cols = image.cols;

            std::vector<float> samples;
            samples.reserve(m_n * m_m); 

            if (image.depth() == CV_8U) 
            {
                for (int i = 0; i < m_m; ++i)
                {
                    for (int j = 0; j < m_n; ++j)
                    {
                        int row = m_random.uniform(0, rows);
                        int col = m_random.uniform(0, cols);
                        uchar point = image.at<uchar>(row, col);
                        samples.push_back(static_cast<float>(point));
                    }
                }
            }
            else if (image.depth() == CV_16U) 
            {
                for (int i = 0; i < m_m; ++i)
                {
                    for (int j = 0; j < m_n; ++j)
                    {
                        int row = m_random.uniform(0, rows);
                        int col = m_random.uniform(0, cols);
                        ushort point = image.at<ushort>(row, col);
                        samples.push_back(static_cast<float>(point));
                    }
                }
            }

            cv::Mat samples_mat(samples); 
            cv::Scalar result = cv::mean(samples_mat); 

            return result[0] * (image.elemSize1() == 1 ? MULT_8_BITS : MULT_16_BITS);
        }

    private:
        cv::RNG m_random;
        int m_n;
        int m_m;

        const double MULT_8_BITS = 1.0 / 255.0;
        const double MULT_16_BITS = 1.0 / 65535.0;
    };
}

