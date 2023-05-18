#pragma once

#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

namespace sky360lib::utils
{
    class Utils
    {
    public:
        static void equalizeImage(const cv::Mat &imageIn, cv::Mat &imageOut, double clipLimit)
        {
            cv::Mat labImage;

            cv::cvtColor(imageIn, labImage, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> labChannels(3);
            cv::split(labImage, labChannels);

            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(clipLimit);
            clahe->setTilesGridSize(cv::Size(6, 6));
            cv::Mat equalizedL;
            clahe->apply(labChannels[0], equalizedL);

            labChannels[0] = equalizedL;
            cv::merge(labChannels, labImage);

            cv::cvtColor(labImage, imageOut, cv::COLOR_YCrCb2BGR);
        }

        // https://stackoverflow.com/questions/6123443/calculating-image-acutance/6129542#6129542
        static double calculateSharpness(cv::Mat& img)
        {
            cv::Mat img_gray;
            if (img.channels() == 3) 
            {
                cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
            } 
            else 
            {
                img_gray = img.clone();
            }

            // Calculate gradients in x and y directions
            cv::Mat grad_x, grad_y;
            cv::Sobel(img_gray, grad_x, CV_64F, 1, 0, 3);
            cv::Sobel(img_gray, grad_y, CV_64F, 0, 1, 3);

            // Calculate gradient magnitude
            cv::Mat grad_mag;
            cv::magnitude(grad_x, grad_y, grad_mag);

            // Calculate mean of gradient magnitude
            cv::Scalar mean = cv::mean(grad_mag);

            return mean[0];
        }
        
        static cv::Mat createHistogram(const cv::Mat &img, int hist_w = 512, int hist_h = 400)
        {
            const int histSize = 256;
            const float range[] = {0, img.elemSize1() == 1 ? 255.0f : 65535.0f};
            const float *histRange = {range};
            const bool uniform = true;
            const bool accumulate = false;

            std::vector<cv::Mat> bgr_planes;
            cv::split(img, bgr_planes);

            cv::Mat b_hist, g_hist, r_hist;
            cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

            int bin_w = cvRound(static_cast<double>(hist_w) / histSize);
            cv::Mat hist_img(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

            cv::normalize(b_hist, b_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
            cv::normalize(g_hist, g_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
            cv::normalize(r_hist, r_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());

            for (int i = 1; i < histSize; ++i)
            {
                cv::line(hist_img,
                         cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                         cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
                         cv::Scalar(255, 0, 0),
                         2,
                         8,
                         0);
                cv::line(hist_img,
                         cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                         cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
                         cv::Scalar(0, 255, 0),
                         2,
                         8,
                         0);
                cv::line(hist_img,
                         cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                         cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
                         cv::Scalar(0, 0, 255),
                         2,
                         8,
                         0);
            }

            return hist_img;
        }

        static std::string formatDouble(double value, int decimal_places = 2) 
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(decimal_places) << value;
            return oss.str();
        }
    };
}