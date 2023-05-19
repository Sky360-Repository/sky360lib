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
        static double estimateSharpness(const cv::Mat& img)
        {
            if (img.channels() == 3) 
            {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }

            // Calculate gradients in x and y directions
            cv::Mat grad_x, grad_y;
            cv::Sobel(img, grad_x, CV_64F, 1, 0, 3);
            cv::Sobel(img, grad_y, CV_64F, 0, 1, 3);

            // Calculate gradient magnitude
            cv::Mat grad_mag;
            cv::magnitude(grad_x, grad_y, grad_mag);

            // Calculate mean of gradient magnitude
            cv::Scalar mean = cv::mean(grad_mag);

            return mean[0];
        }

        // Based on: https://www.sciencedirect.com/science/article/abs/pii/S1077314296900600
        static double estimateNoise(const cv::Mat img) 
        {
            if (img.channels() == 3) 
            {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }

            cv::Mat laplacianMask = (cv::Mat_<double>(3,3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);

            cv::Mat laplacianImage;
            cv::filter2D(img, laplacianImage, -1, laplacianMask, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

            double sigma = cv::sum(cv::abs(laplacianImage))[0];

            int H = img.rows - 2;
            int W = img.cols - 2;
            sigma = sigma * sqrt(0.5 * M_PI) / (6.0 * W * H);

            return sigma;
        }

        // Based on "Noise Aware Image Assessment metric based Auto Exposure Control" by "Uk Cheol Shin, KAIST RCV LAB"
        // Can be used to quantify the amount of information, or "texture", in an image.
        // Normalised here so 1 represents maximum entropy (an image with a perfectly uniform histogram, meaning each gray level is equally probable)
        // and 0 represents minimum entropy (an image where every pixel has the same color).
        static float estimateEntropy(const cv::Mat img)
        {
            if (img.channels() == 3) 
            {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }

            cv::Mat hist;
            const int histSize = 256;

            // Compute the histograms:
            float range[] = {0, histSize};
            const float *histRange = {range};

            // images, number of images, channels, mask, hist, dim, histsize, ranges,uniform, accumulate
            cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

            // compute entropy
            float entropy_value = 0;
            float total_size = img.rows * img.cols; // total size of all symbols in an image

            float *sym_occur = hist.ptr<float>(0); // the number of times a sybmol has occured
            for (int i = 0; i < histSize; i++)
            {
                if (sym_occur[i] > 0) // log of zero goes to infinity
                {
                    entropy_value += (sym_occur[i] / total_size) * (std::log2(total_size / sym_occur[i]));
                }
            }

            entropy_value /= 8.0; // the max entropy for an 8-bit grayscale image is 8, so needs to be adjusted for 16

            hist.release();

            return entropy_value;
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