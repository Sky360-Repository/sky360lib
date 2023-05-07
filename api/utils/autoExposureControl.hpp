#pragma once

#include <opencv2/opencv.hpp>

namespace sky360lib::utils
{
    class AutoExposureControl
    {
    public:
        struct ExposureAdjustment
        {
            double exposure;
            double gain;
        };

        AutoExposureControl(double day_targetMSV = 1.8, 
                            double night_targetMSV = 0.98,
                            double min_exposure = 100, 
                            double max_exposure = 50000, 
                            double min_gain = 5, 
                            double max_gain = 20, 
                            double max_exposure_step = 4000)
        : targetMSV_(day_targetMSV)
        , day_targetMSV_(day_targetMSV)
        , night_targetMSV_(night_targetMSV)
        , min_exposure_(min_exposure)
        , max_exposure_(max_exposure)
        , min_gain_(min_gain)
        , max_gain_(max_gain)
        , max_exposure_step_(max_exposure_step)
        , err_i_(0.0)
        , is_night_(false)
        {}

        double get_targetMSV() const { return targetMSV_; }
        void set_targetMSV(double targetMSV) { targetMSV_ = std::clamp(targetMSV, 0.9, 5.0); }

        double get_min_exposure() const { return min_exposure_; }
        void set_min_exposure(double min_exposure) { min_exposure_ = min_exposure; }

        double get_max_exposure() const { return max_exposure_; }
        void set_max_exposure(double max_exposure) { max_exposure_ = max_exposure; }

        double get_min_gain() const { return min_gain_; }
        void set_min_gain(double min_gain) { min_gain_ = min_gain; }

        double get_max_gain() const { return max_gain_; }
        void set_max_gain(double max_gain) { max_gain_ = max_gain; }

        double get_max_exposure_step() const { return max_exposure_step_; }
        void set_max_exposure_step(double max_exposure_step) { max_exposure_step_ = max_exposure_step; }

        bool is_day() const { return !is_night_; }

        void toggle_day_night()
        {
            is_night_ = !is_night_;
            targetMSV_ = is_night_ ? night_targetMSV_ : day_targetMSV_;
        }

        ExposureAdjustment calculate_exposure_gain(const cv::Mat &cv_image, double current_exposure, double current_gain)
        {
            cv::Mat brightness_image;

            if (cv_image.channels() == 3)
            {
                cv::cvtColor(cv_image, brightness_image, cv::COLOR_BGR2HSV);
                std::vector<cv::Mat> hsv_channels;
                cv::split(brightness_image, hsv_channels);
                brightness_image = hsv_channels[2];
            }
            else
            {
                brightness_image = cv_image;
            }

            cv::Mat hist;
            cv::calcHist(std::vector<cv::Mat>{brightness_image}, {0}, cv::Mat(), hist, {5}, {0, cv_image.elemSize1() == 1 ? 255.0f : 65535.0f});

            double mean_sample_value = 0;
            for (int i = 0; i < hist.rows; ++i)
            {
                mean_sample_value += hist.at<float>(i) * (i + 1);
            }

            mean_sample_value /= cv_image.size().area();

            // Proportional and integral constants (k_p and k_i)
            const double k_p = 400;
            const double k_i = 80;
            const double max_i = 3;

            const double err_p = targetMSV_ - mean_sample_value;

            err_i_ += err_p;

            if (std::abs(err_i_) > max_i)
            {
                err_i_ = std::copysign(max_i, err_i_);
            }

            if (std::abs(err_p) > 0.15) // To get a stable exposure
            {
                double new_exposure, new_gain;

                // Decreasing gain and exposure
                if (err_p < 0)
                {
                    if (current_gain > min_gain_)
                    {
                        const double gain_decrement = std::abs(err_p) * 1.1;
                        new_gain = std::min(std::max(current_gain - gain_decrement, min_gain_), max_gain_);
                    }
                    else
                    {
                        new_gain = current_gain;
                    }

                    // Decrease exposure if the gain has reached its minimum value
                    if (new_gain == current_gain)
                    {
                        const double desired_exposure_change = k_p * err_p + k_i * err_i_;
                        const double exposure_change = std::clamp(desired_exposure_change, -max_exposure_step_, max_exposure_step_);
                        new_exposure = current_exposure + exposure_change;
                    }
                    else
                    {
                        new_exposure = current_exposure;
                    }

                    if (new_exposure <= min_exposure_ && new_gain == current_gain)
                    {
                        new_exposure = min_exposure_;
                        toggle_day_night();
                    }

                    return { .exposure = new_exposure, .gain = new_gain };
                }


                // Calculate the desired exposure change
                const double desired_exposure_change = k_p * err_p + k_i * err_i_;

                // Limit the exposure change to max_exposure_step_
                const double exposure_change = std::clamp(desired_exposure_change, -max_exposure_step_, max_exposure_step_);

                // Update the exposure value
                new_exposure = current_exposure + exposure_change;

                // Increasing gain and exposure
                if (new_exposure > max_exposure_)
                {
                    new_exposure = max_exposure_;
                    const double gain_increment = (targetMSV_ - mean_sample_value) * 1.1;
                    new_gain = std::min(std::max(current_gain + gain_increment, min_gain_), max_gain_);

                    // If we are unable to set exposure or gain anymore to achieve the target brightness, toggle between day and night
                    if (new_gain == current_gain)
                    {
                        toggle_day_night();
                    }

                    return { .exposure = new_exposure, .gain = new_gain };
                }

                return { .exposure = new_exposure, .gain = current_gain };
            }
            else
            {
                return { .exposure = current_exposure, .gain = current_gain };
            }
        }

    private:
        double targetMSV_;
        double day_targetMSV_;
        double night_targetMSV_;
        double min_exposure_;
        double max_exposure_;
        double min_gain_;
        double max_gain_;
        double max_exposure_step_;
        double err_i_;
        bool is_night_;
    };
}
