#pragma once

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
            clahe->setTilesGridSize(cv::Size(32, 32));
            cv::Mat equalizedL;
            clahe->apply(labChannels[0], equalizedL);

            labChannels[0] = equalizedL;
            cv::merge(labChannels, labImage);

            cv::cvtColor(labImage, imageOut, cv::COLOR_YCrCb2BGR);
        }

        static cv::Mat createHistogram(const cv::Mat &img, int hist_w = 512, int hist_h = 400)
        {
            // Set up the parameters for the histogram
            int histSize = 256;
            float range[] = {0, img.elemSize1() == 1 ? 256.0f : 65536.0f};
            const float *histRange = {range};
            bool uniform = true;
            bool accumulate = false;

            // Calculate the histograms for each channel
            std::vector<cv::Mat> bgr_planes;
            cv::split(img, bgr_planes);

            cv::Mat b_hist, g_hist, r_hist;
            cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

            // Create an image for the histogram
            int bin_w = cvRound(static_cast<double>(hist_w) / histSize);
            cv::Mat hist_img(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

            // Normalize the histograms for better visualization
            cv::normalize(b_hist, b_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
            cv::normalize(g_hist, g_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
            cv::normalize(r_hist, r_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());

            // Draw the histograms
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
    };
}