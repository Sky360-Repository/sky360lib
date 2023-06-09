#pragma once

#include <opencv2/opencv.hpp>

namespace sky360lib::utils
{
    struct AutoExposureValues
    {
        double exposure;
        double gain;
    };

    class AutoExposure
    {
    public:
        AutoExposure(double target_msv = 0.25, 
                    double min_exposure = 100.0, 
                    double max_exposure = 50000.0, 
                    double min_gain = 0.0, 
                    double max_gain = 25.0, 
                    double step_gain = 1.0,
                    double max_exposure_step = 4000.0,
                    double default_exposure = 20000,
                    double default_gain = 5)
            : m_target_msv(target_msv)
            , m_min_target_msv(0.02)
            , m_max_target_msv(0.25)
            , m_current_msv(0.0)
            , m_min_exposure(min_exposure)
            , m_max_exposure(max_exposure)
            , m_min_gain(min_gain)
            , m_max_gain(max_gain)
            , m_step_gain(step_gain)
            , m_max_exposure_step(max_exposure_step)
            , m_err_i(0.0)
            , m_is_night(false)
            , m_params{default_exposure, default_gain}
        {}

        double get_target_msv() const { return m_target_msv; }
        void set_target_msv(double target_msv) { m_target_msv = std::clamp(target_msv, m_min_target_msv, m_max_target_msv); }

        bool is_day() const { return !m_is_night; }
        double get_current_msv() const { return m_current_msv; }

        AutoExposureValues getParams() const { return m_params; }

        void process(cv::Mat &cv_image, const cv::InputArray & mask = cv::noArray())
        {
            cv::Mat grayscale_image;
            if (cv_image.channels() == 3)
            {
                cv::cvtColor(cv_image, grayscale_image, cv::COLOR_BGR2GRAY);
            }
            else
            {
                grayscale_image = cv_image;
            }

            auto illumination = illumination_estimation(grayscale_image, 50, 50);
            m_current_msv = illumination.first * (cv_image.elemSize1() == 1 ? MULT_8_BITS : MULT_16_BITS);

            const double maxStepA = 5;
            const double maxStepB = 4000;

            double normalizedExposure = (m_params.exposure - m_max_exposure) / (m_min_exposure - m_max_exposure);
            m_max_exposure_step = maxStepA + (maxStepB - maxStepA) * (1 - pow(normalizedExposure, 0.05));
            
            m_is_night = (m_params.exposure >= m_max_exposure);

            const double k_p = 400;
            const double k_i = 80;
            const double max_i = 1.5;
            const double err_p = m_target_msv - m_current_msv;
            m_err_i += err_p;

            if (std::abs(m_err_i) > max_i)
            {
                m_err_i = std::copysign(max_i, m_err_i);
            }

            if (std::abs(err_p) > 0.025) // To get a stable exposure
            {

                // When light is decreasing
                if (err_p > 0)
                {
                    // Increase exposure first
                    if (m_params.exposure < m_max_exposure)
                    {
                        const double desired_exposure_change = k_p * err_p + k_i * m_err_i;
                        const double exposure_change = std::clamp(desired_exposure_change, -m_max_exposure_step, m_max_exposure_step);
                        m_params.exposure = m_params.exposure + exposure_change;
                        return;
                    }

                    // Then decrease target_msv
                    if (m_target_msv > m_min_target_msv)
                    {
                        m_target_msv -= 0.01;
                        m_target_msv = std::clamp(m_target_msv, m_min_target_msv, m_max_target_msv);
                        return;
                    }

                    // Finally increase gain
                    if (m_params.gain < m_max_gain)
                    {
                        m_params.gain = std::clamp(m_params.gain + m_step_gain, m_min_gain, m_max_gain);
                        return;
                    }
                }

                // When light is increasing
                else if (err_p < 0)
                {
                    // Decrease gain first
                    if (m_params.gain > m_min_gain)
                    {
                        m_params.gain = std::clamp(m_params.gain - m_step_gain, m_min_gain, m_max_gain);
                        return;
                    }

                    // Then increase target_msv
                    if (m_target_msv < m_max_target_msv)
                    {
                        m_target_msv += 0.01;
                        m_target_msv = std::clamp(m_target_msv, m_min_target_msv, m_max_target_msv);
                        return;
                    }

                    // Finally decrease exposure
                    if (m_params.exposure > m_min_exposure)
                    {
                        const double desired_exposure_change = k_p * err_p + k_i * m_err_i;
                        const double exposure_change = std::clamp(desired_exposure_change, -m_max_exposure_step, m_max_exposure_step);
                        m_params.exposure = m_params.exposure + exposure_change;
                        return;
                    }
                }
            }

            return;
        }

    private:
        std::pair<double, double> illumination_estimation(cv::Mat& image, int n, int m)
        {
            int rows = image.rows;
            int cols = image.cols;

            std::vector<float> samples;
            samples.reserve(n * m);  // allocate memory in advance

            if (image.depth() == CV_8U) 
            {
                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < n; ++j)
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
                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        int row = m_random.uniform(0, rows);
                        int col = m_random.uniform(0, cols);
                        ushort point = image.at<ushort>(row, col);
                        samples.push_back(static_cast<float>(point));
                    }
                }
            }

            cv::Mat samples_mat(samples); 
            cv::Mat mean, stddev;
            cv::meanStdDev(samples_mat, mean, stddev); // Could use Std Dev for under or over exposure estimation

            return { mean.at<double>(0), stddev.at<double>(0) };
        }

        const double MULT_8_BITS = 1.0 / 255.0;
        const double MULT_16_BITS = 1.0 / 65535.0;

        double m_target_msv;
        double m_min_target_msv;
        double m_max_target_msv;
        double m_current_msv;
        double m_min_exposure;
        double m_max_exposure;
        double m_min_gain;
        double m_max_gain;
        double m_step_gain;
        double m_max_exposure_step;
        double m_err_i;
        bool m_is_night;
        AutoExposureValues m_params;
        cv::RNG m_random;
    };
}