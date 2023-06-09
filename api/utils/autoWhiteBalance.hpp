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

        AutoWhiteBalance()
            : m_default_wb({165.0, 128.0, 240.0})
            , m_current_wb(m_default_wb)
            , m_error_threshold(0.25)
        {
        }

        // Estimates the global scene illumination using subsampling approach over m x n samples
        // https://ipg.fer.hr/_download/repository/Improving_the_White-patch_method_by_subsampling.pdf
        WhiteBalanceValues illumination_estimation(cv::Mat& image, int n, int m)
        {
            const double MAX_VALUE_8_BIT = 255.0;
            const double MAX_VALUE_16_BIT = 65535.0;

            int rows = image.rows;
            int cols = image.cols;
            cv::Scalar result(0.0, 0.0, 0.0);

            if (image.depth() == CV_8U) 
            {
                for (int i = 0; i < m; ++i)
                {
                    cv::Scalar max(0.0, 0.0, 0.0);
                    for (int j = 0; j < n; ++j)
                    {
                        int row = random.uniform(0, rows);
                        int col = random.uniform(0, cols);
                        cv::Vec3b point = image.at<cv::Vec3b>(row, col); 

                        for (int k = 0; k < 3; ++k)
                        {
                            max[k] = std::max(static_cast<double>(max[k]), static_cast<double>(point[k]));
                        }
                    }
                    result += max;
                }
            }
            else if (image.depth() == CV_16U) 
            {
                for (int i = 0; i < m; ++i)
                {
                    cv::Scalar max(0.0, 0.0, 0.0);
                    for (int j = 0; j < n; ++j)
                    {
                        int row = random.uniform(0, rows);
                        int col = random.uniform(0, cols);
                        cv::Vec3w point = image.at<cv::Vec3w>(row, col);

                        for (int k = 0; k < 3; ++k)
                        {
                            max[k] = std::max(static_cast<double>(max[k]), static_cast<double>(point[k]));
                        }
                    }
                    result += max;
                }
            }

            double sum = result.dot(result) / 3.0;
            sum = sqrt(sum);
            result /= sum;

            cv::Scalar unit_vec(1.0, 1.0, 1.0);
            double error = cv::norm(result - unit_vec);
            
            double max_val = image.depth() == CV_8U ? MAX_VALUE_8_BIT : MAX_VALUE_16_BIT;

            if (error > m_error_threshold) 
            {
                WhiteBalanceValues wb_values = 
                { 
                    std::max(0.0, std::min(max_val, (1.0 / result[2]) * m_current_wb.red)),
                    std::max(0.0, std::min(max_val, (1.0 / result[1]) * m_current_wb.green)),
                    std::max(0.0, std::min(max_val, (1.0 / result[0]) * m_current_wb.blue))
                };
                m_current_wb = wb_values; 
            }

            return m_current_wb;
        }

        WhiteBalanceValues getDefaultWhiteBalance() const
        {
            return m_default_wb;
        }

    private:
        const WhiteBalanceValues m_default_wb;
        WhiteBalanceValues m_current_wb;
        const double m_error_threshold;
        cv::RNG random;
    };
}