#pragma once

#include <opencv2/opencv.hpp>

namespace sky360lib::utils{

    class AutoExposureControl {
    public:
        AutoExposureControl(double targetMSV = 1.5, double max_exposure = 50000, double min_gain = 5, double max_exposure_step = 4000)
            : targetMSV_(targetMSV), max_exposure_(max_exposure), min_gain_(min_gain), max_exposure_step_(max_exposure_step), err_i_(0.0) {}

        double get_targetMSV() const { return targetMSV_; }
        void set_targetMSV(double targetMSV) { targetMSV_ = targetMSV; }

        double get_max_exposure() const { return max_exposure_; }
        void set_max_exposure(double max_exposure) { max_exposure_ = max_exposure; }

        double get_min_gain() const { return min_gain_; }
        void set_min_gain(double min_gain) { min_gain_ = min_gain; }

        double get_max_exposure_step() const { return max_exposure_step_; }
        void set_max_exposure_step(double max_exposure_step) { max_exposure_step_ = max_exposure_step; }

        double calculate_brightness(const cv::Mat &frame)
        {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::Scalar mean_scalar = cv::mean(gray);
            double brightness = mean_scalar[0];
            return brightness;
        }

        std::pair<double, double> calculate_exposure_gain(const cv::Mat& cv_image, double current_exposure, double current_gain)
        {
            int rows = cv_image.rows;
            int cols = cv_image.cols;
            int channels = cv_image.channels();
            cv::Mat brightness_image;

            if (channels == 3)
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
            cv::calcHist(std::vector<cv::Mat>{brightness_image}, {0}, cv::Mat(), hist, {5}, {0, 256});

            double mean_sample_value = 0;
            for (int i = 0; i < hist.rows; ++i)
            {
                mean_sample_value += hist.at<float>(i) * (i + 1);
            }

            mean_sample_value /= (rows * cols);

            // Proportional and integral constants (k_p and k_i)
            double k_p = 400;
            double k_i = 80;
            double max_i = 3;

            double err_p = targetMSV_ - mean_sample_value;

            err_i_ += err_p;

            if (std::abs(err_i_) > max_i)
            {
                err_i_ = std::copysign(max_i, err_i_);
            }

            if (std::abs(err_p) > 0.2) // To get a stable exposure 
            {
                double new_exposure, new_gain;
                if (err_p < 0 && current_gain > min_gain_)
                {
                    double gain_decrement = std::abs(err_p) * 1.1;
                    new_gain = std::max(current_gain - gain_decrement, min_gain_);

                    return std::make_pair(current_exposure, new_gain);
                }

                // Calculate the desired exposure change
                double desired_exposure_change = k_p * err_p + k_i * err_i_;

                // Limit the exposure change to max_exposure_step_
                double exposure_change = std::clamp(desired_exposure_change, -max_exposure_step_, max_exposure_step_);

                // Update the exposure value
                new_exposure = current_exposure + exposure_change;

                if (new_exposure > max_exposure_)
                {
                    new_exposure = max_exposure_;
                    double gain_increment = (targetMSV_ - mean_sample_value) * 1.1;
                    new_gain = std::max(current_gain + gain_increment, min_gain_);

                    return std::make_pair(new_exposure, new_gain);
                }

                return std::make_pair(new_exposure, current_gain);
            }
            else
            {
                return std::make_pair(current_exposure, current_gain);
            }
        }

    private:
        double targetMSV_;
        double max_exposure_;
        double min_gain_;
        double max_exposure_step_;
        double err_i_;
    };

}
