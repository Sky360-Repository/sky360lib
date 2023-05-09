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

        AutoExposureControl(double day_target_msv = 0.24, 
                            double night_target_msv = 0.05,
                            double min_exposure = 100, 
                            double max_exposure = 50000, 
                            double min_gain = 1.0, 
                            double max_gain = 30.0, 
                            double step_gain = 1.0,
                            double max_exposure_step = 4000)
        : target_msv_(day_target_msv)
        , day_target_msv_(day_target_msv)
        , night_target_msv_(night_target_msv)
        , current_msv_(0.0)
        , min_exposure_(min_exposure)
        , max_exposure_(max_exposure)
        , min_gain_(min_gain)
        , max_gain_(max_gain)
        , step_gain_(step_gain)
        , max_exposure_step_(max_exposure_step)
        , err_i_(0.0)
        , is_night_(false)
        {} 

        double get_target_msv() const { return target_msv_; }
        void set_target_msv(double target_msv) { target_msv_ = std::clamp(target_msv, 0.18, 1.0); }

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

        double get_current_msv() const { return current_msv_; }

        ExposureAdjustment calculate_exposure_gain(const cv::Mat &cv_image, double current_exposure, double current_gain)
        {
            const static double MULT_8_BITS = 1.0 / 255.0;
            const static double MULT_16_BITS = 1.0 / 65535.0;

            cv::Mat brightness_image;

            if (cv_image.channels() == 3)
            {
                cv::cvtColor(cv_image, brightness_image, cv::COLOR_BGR2YCrCb);
                std::vector<cv::Mat> hsv_channels;
                cv::split(brightness_image, hsv_channels);
                brightness_image = hsv_channels[0];
            }
            else
            {
                brightness_image = cv_image;
            }

            cv::Scalar mean_intensity = cv::mean(brightness_image);
            current_msv_ = mean_intensity[0] * (cv_image.elemSize1() == 1 ? MULT_8_BITS : MULT_16_BITS);
            
            // switching mechanism
            if (current_exposure >= max_exposure_ && current_gain >= max_gain_)
            {
                is_night_ = true;
                target_msv_ = night_target_msv_;
            }
            else if (current_exposure <= min_exposure_ && current_gain <= min_gain_)
            {
                is_night_ = false;
                target_msv_ = day_target_msv_;
            }

            // Proportional and integral constants (k_p and k_i)
            const double k_p = 400;
            const double k_i = 80;
            const double max_i = 3;

            const double err_p = target_msv_ - current_msv_;

            err_i_ += err_p;

            if (std::abs(err_i_) > max_i)
            {
                err_i_ = std::copysign(max_i, err_i_);
            }
            // std::cout << "err_p: " << err_p << ", err_i: " << err_i_ << std::endl;

            if (std::abs(err_p) > 0.01) // To get a stable exposure
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
                    }
                    //std::cout << "decrease: new_exposure: " << new_exposure << ", new_gain: " << new_gain << std::endl;

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
                    new_gain = std::min(std::max(current_gain + step_gain_, min_gain_), max_gain_);

                    // std::cout << "increase gain: new_exposure: " << new_exposure << ", new_gain: " << new_gain << std::endl;
                    return { .exposure = new_exposure, .gain = new_gain };
                }

                // std::cout << "increase exposure: new_exposure: " << new_exposure << ", new_gain: " << current_gain << std::endl;
                return { .exposure = new_exposure, .gain = current_gain };
            }
            else
            {
                //std::cout << "dont change: new_exposure: " << current_exposure << ", new_gain: " << current_gain << std::endl;
                return { .exposure = current_exposure, .gain = current_gain };
            }
        }

    private:
        double target_msv_;
        double day_target_msv_;
        double night_target_msv_;
        double current_msv_;
        double min_exposure_;
        double max_exposure_;
        double min_gain_;
        double max_gain_;
        double step_gain_;
        double max_exposure_step_;
        double err_i_;
        bool is_night_;
    };
}
