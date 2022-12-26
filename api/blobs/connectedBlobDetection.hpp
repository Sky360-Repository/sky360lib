#pragma once

#include "coreUtils.hpp"

#include <opencv2/core.hpp>

namespace sky360lib::blobs
{
    class ConnectedBlobDetection final
    {
    public:

        /// Detects the number of available threads to use
        /// Will set the number fo threads to the number of avaible threads - 1
        static const size_t DETECT_NUMBER_OF_THREADS{0};

        ConnectedBlobDetection(size_t _numProcessesParallel = DETECT_NUMBER_OF_THREADS);

        // Finds the connected components in the image and returns a list of bounding boxes
        bool detect(const cv::Mat &image, std::vector<cv::Rect>& bboxes);

    private:
        size_t m_numProcessesParallel;
        bool m_initialized;
        cv::Mat m_labels;
        std::vector<size_t> m_processSeq;
        std::vector<std::unique_ptr<ImgSize>> m_imgSizesParallel;
        std::vector<std::vector<cv::Rect>> m_bboxesParallel;

        void prepareParallel(const cv::Mat &_image);
        static void applyDetect(const cv::Mat& _labels, std::vector<cv::Rect> &_bboxes);
    };
}