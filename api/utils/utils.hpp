#pragma once

#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "ringbuf.h"
#include "textWriter.hpp"

namespace sky360lib::utils
{
    class Utils
    {
    public:
        static void equalize_image(const cv::Mat &imageIn, cv::Mat &imageOut, double clipLimit)
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
        static double estimate_sharpness(const cv::Mat &img)
        {
            if (img.empty())
            {
                return 0.0;
            }

            cv::Mat sharpness_img;
            if (img.channels() == 3)
            {
                cv::cvtColor(img, sharpness_img, cv::COLOR_BGR2GRAY);
            }
            else
            {
                sharpness_img = img;
            }

            // Calculate gradients in x and y directions
            cv::Mat grad_x, grad_y;
            cv::Sobel(sharpness_img, grad_x, CV_64F, 1, 0, 3);
            cv::Sobel(sharpness_img, grad_y, CV_64F, 0, 1, 3);

            // Calculate gradient magnitude
            cv::Mat grad_mag;
            cv::magnitude(grad_x, grad_y, grad_mag);

            // Calculate mean of gradient magnitude
            cv::Scalar mean = cv::mean(grad_mag);

            return mean[0];
        }

        // Based on: https://www.sciencedirect.com/science/article/abs/pii/S1077314296900600
        static double estimate_noise(const cv::Mat &img)
        {
            if (img.empty())
            {
                return 0.0;
            }

            cv::Mat noise_img;
            if (img.channels() == 3)
            {
                cv::cvtColor(img, noise_img, cv::COLOR_BGR2GRAY);
            }
            else
            {
                noise_img = img;
            }

            const cv::Mat laplacianMask = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);

            cv::Mat laplacianImage;
            cv::filter2D(noise_img, laplacianImage, -1, laplacianMask, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

            double noise_height = noise_img.rows - 2;
            double noise_width = noise_img.cols - 2;
            double sigma = cv::sum(cv::abs(laplacianImage))[0] * std::sqrt(0.5 * M_PI) / (6.0 * noise_width * noise_height);

            return sigma;
        }

        // Based on "Noise Aware Image Assessment metric based Auto Exposure Control" by "Uk Cheol Shin, KAIST RCV LAB"
        // Can be used to quantify the amount of information, or "texture", in an image.
        // Normalised here so 1 represents maximum entropy (an image with a perfectly uniform histogram, meaning each gray level is equally probable)
        // and 0 represents minimum entropy (an image where every pixel has the same color).
        static float estimate_entropy(const cv::Mat &img)
        {
            if (img.empty())
            {
                return 0.0;
            }

            cv::Mat entropy_img;
            if (img.channels() == 3)
            {
                cv::cvtColor(img, entropy_img, cv::COLOR_BGR2GRAY);
            }
            else
            {
                entropy_img = img;
            }

            cv::Mat hist;
            const int histSize = 256;

            // Compute the histograms:
            const float range[] = {0, histSize};
            const float *histRange = {range};

            // images, number of images, channels, mask, hist, dim, histsize, ranges,uniform, accumulate
            cv::calcHist(&entropy_img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

            // compute entropy
            double entropy_value = 0;
            const double total_size = entropy_img.rows * entropy_img.cols; // total size of all symbols in an image

            float *sym_occur = hist.ptr<float>(0); // the number of times a sybmol has occured
            for (int i = 0; i < histSize; ++i)
            {
                if (sym_occur[i] > 0) // log of zero goes to infinity
                {
                    entropy_value += ((double)sym_occur[i] / total_size) * (std::log2(total_size / (double)sym_occur[i]));
                }
            }

            entropy_value /= 8.0; // the max entropy for an 8-bit grayscale image is 8, so needs to be adjusted for 16

            hist.release();

            return entropy_value;
        }

        static cv::Mat create_histogram(const cv::Mat &img, int hist_w = 512, int hist_h = 400)
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

        template<typename Container>
        static cv::Mat draw_graph(const std::string& name, const Container &data, const cv::Size &graphSize, int type, cv::Scalar lineColor, cv::Scalar rectColor)
        {
            const TextWriter text_writter(cv::Scalar{255, 255, 255, 0}, 9, 2.5); 
            const TextWriter text_writter_name(lineColor, 6, 3.5); 
            cv::Mat graph(graphSize, type);

            cv::rectangle(graph, cv::Rect(0, 0, graphSize.width, graphSize.height), rectColor, cv::FILLED);

            const double minVal = *std::min_element(data.begin(), data.end());
            const double maxVal = *std::max_element(data.begin(), data.end());
            const double min_graph = minVal * 0.9;
            const double max_graph = maxVal * 1.1;
            const double normalization_mult = 1.0 / (max_graph - min_graph) * graphSize.height;

            for (size_t i = 1; i < data.size(); ++i)
            {
                const double val0 = (data[i - 1] - min_graph) * normalization_mult;
                const double val1 = (data[i] - min_graph) * normalization_mult;
                cv::line(graph, cv::Point(i - 1, graphSize.height - val0), cv::Point(i, graphSize.height - val1), lineColor);
            }

            text_writter_name.writeText(graph, name, 1, false);
            text_writter.writeText(graph, format_double(maxVal, 3), 1, true);
            text_writter.writeText(graph, format_double(minVal, 3), 8, true);
            text_writter_name.writeText(graph, format_double(data.back(), 3), 6, false);

            return graph;
        }

        static void overlay_mage(cv::Mat &dst, const cv::Mat &src, cv::Point location, double alpha)
        {
            cv::Mat overlay;
            dst.copyTo(overlay);

            cv::Rect roi(location.x, location.y, src.cols, src.rows);
            cv::Mat subImage = overlay(roi);

            src.copyTo(subImage);

            cv::addWeighted(overlay, alpha, dst, 1 - alpha, 0.0, dst);
        }

        static std::string format_double(double value, int decimal_places = 2)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(decimal_places) << value;
            return oss.str();
        }
    };
}