#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace sky360lib::utils
{
    class SubSampler 
    {
    public:
        SubSampler(int n, int m)
            : m_n(n), m_m(m) {}

        cv::Mat subSample(const cv::Mat& image)
        {
            const int rows = image.rows;
            const int cols = image.cols;

            cv::Mat subSampled(rows / m_m, cols / m_n, image.type());

            if (image.channels() == 1) // Grayscale image
            {
                for (int i = 0; i < subSampled.rows; ++i)
                {
                    for (int j = 0; j < subSampled.cols; ++j)
                    {
                        const int row = i * m_m;
                        const int col = j * m_n;
                        subSampled.at<uchar>(i, j) = image.at<uchar>(row, col);
                    }
                }
            }
            else if (image.channels() == 3) // Color image
            {
                for (int i = 0; i < subSampled.rows; ++i)
                {
                    for (int j = 0; j < subSampled.cols; ++j)
                    {
                        const int row = i * m_m;
                        const int col = j * m_n;
                        subSampled.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(row, col);
                    }
                }
            }
            else
            {
                // Handle unsupported image type/error condition
            }

            return subSampled;
        }

    private:
        int m_n;
        int m_m;
    };
}