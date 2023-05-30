#pragma once

#include <opencv2/opencv.hpp>

namespace sky360lib::utils 
{
    class AutoWhiteBalance 
    {
    public:
        struct WhiteBalanceValues 
        {
            double red;
            double green;
            double blue;

            bool apply;
        };

        AutoWhiteBalance(double max_exposure, double adjustment_factor = 0.9, 
            double threshold = 3.0, double exposure_change_threshold = 0.15)
            : m_adjustment_factor(adjustment_factor)
            , m_threshold(threshold)
            , m_exposure_change_threshold(exposure_change_threshold)
            , m_max_exposure(max_exposure)
            , m_default_wb({165.0, 128.0, 240.0, true})
            , m_current_wb(m_default_wb)
            , m_previous_exposure(-1.0)
            , m_use_generic_white_balance(true)
        {
        }

        WhiteBalanceValues gray_world(const cv::Mat& image, double current_exposure)
        {
            auto do_calc = do_calc_wb(current_exposure);
            if (do_calc == Generic)
            {
                return m_default_wb;
            }
            else if (do_calc == DontChange)
            {
                return {0.0, 0.0, 0.0, false};
            }

            m_previous_exposure = current_exposure;

            const cv::Scalar imgMean = cv::mean(image);

            const double avg = (imgMean[0] + imgMean[1] + imgMean[2]) / 3.0;

            WhiteBalanceValues wb_values = 
            { 
                std::max(0.0, std::min(255.0, m_current_wb.red * (1.0 + m_adjustment_factor * (avg / imgMean[2] - 1.0)))),
                std::max(0.0, std::min(255.0, m_current_wb.green * (1.0 + m_adjustment_factor * (avg / imgMean[1] - 1.0)))),
                std::max(0.0, std::min(255.0, m_current_wb.blue * (1.0 + m_adjustment_factor * (avg / imgMean[0] - 1.0)))),
                true
            };

            // Check if the changes are substantial enough
            if (abs(wb_values.red - m_current_wb.red) > m_threshold ||
                abs(wb_values.green - m_current_wb.green) > m_threshold ||
                abs(wb_values.blue - m_current_wb.blue) > m_threshold) 
            {
                m_current_wb = wb_values;
            } 

            return m_current_wb;
        }

        static WhiteBalanceValues grayWorld(const cv::Mat& image, const WhiteBalanceValues& currentWB, double adjustmentFactor, double threshold) 
        {
            const cv::Scalar imgMean = cv::mean(image);

            const double avg = (imgMean[0] + imgMean[1] + imgMean[2]) / 3.0;

            WhiteBalanceValues wbValues = 
            { 
                std::max(0.0, std::min(255.0, currentWB.red * (1.0 + adjustmentFactor * (avg / imgMean[2] - 1.0)))),
                std::max(0.0, std::min(255.0, currentWB.green * (1.0 + adjustmentFactor * (avg / imgMean[1] - 1.0)))),
                std::max(0.0, std::min(255.0, currentWB.blue * (1.0 + adjustmentFactor * (avg / imgMean[0] - 1.0)))),
                true
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

    private:
        const double m_adjustment_factor;
        const double m_threshold;
        const double m_exposure_change_threshold;
        const double m_max_exposure;
        const WhiteBalanceValues m_default_wb;
        WhiteBalanceValues m_current_wb;
        double m_previous_exposure;
        bool m_use_generic_white_balance;

        enum WBCalc
        {
            Calculate,
            Generic,
            DontChange
        };

        WBCalc do_calc_wb(double current_exposure)
        {
            if (m_previous_exposure < 0.0)
            {
                m_previous_exposure = current_exposure;
            }
            const double exposure_change = std::abs(current_exposure - m_previous_exposure) / m_previous_exposure;
            if (exposure_change >= m_exposure_change_threshold && current_exposure < m_max_exposure)
            {
                m_use_generic_white_balance = true;
                return Calculate;
            }
            else if (current_exposure >= m_max_exposure && m_use_generic_white_balance)
            {
                m_use_generic_white_balance = false;
                return Generic;
            }

            return DontChange;
        }
    };
} 
