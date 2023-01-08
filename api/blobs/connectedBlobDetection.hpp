#pragma once

#include "coreUtils.hpp"

#include <opencv2/core.hpp>

namespace sky360lib::blobs
{
    class ConnectedBlobDetection final
    {
    public:
        static const int DEFAULT_SIZE_THRESHOLD = 5;
        static const int DEFAULT_AREA_THRESHOLD = 25;

        /// Detects the number of available threads to use
        /// Will set the number fo threads to the number of avaible threads - 1
        static const size_t DETECT_NUMBER_OF_THREADS{0};

        ConnectedBlobDetection(size_t _numProcessesParallel = DETECT_NUMBER_OF_THREADS);

        // Finds the connected components in the image and returns a list of bounding boxes
        bool detect(const cv::Mat &_image, std::vector<cv::Rect>& _bboxes);

        bool detectOld(const cv::Mat &_image, std::vector<cv::Rect> &_bboxes);

        inline void setSizeThreshold(int _threshold) { m_sizeThreshold = std::max(_threshold, 2); }
        inline void setAreaThreshold(int _threshold) { m_sizeThreshold = std::max(_threshold, 4); }

        // Finds the connected components in the image and returns a list of keypoints
        // This function uses detect and converts from Rect to KeyPoints using a fixed scale
        std::vector<cv::KeyPoint> detectKP(const cv::Mat &_image);

        // Finds the connected components in the image and returns a list of bounding boxes
        std::vector<cv::Rect> detectRet(const cv::Mat &_image);

    private:
        int m_sizeThreshold;
        int m_areaThreshold;
        size_t m_numProcessesParallel;
        bool m_initialized;
        cv::Mat m_labels;
        std::vector<size_t> m_processSeq;
        std::vector<std::unique_ptr<ImgSize>> m_imgSizesParallel;
        std::vector<std::vector<cv::Rect>> m_bboxesParallel;

        void prepareParallel(const cv::Mat &_image);
        static void applyDetectBBoxes(const cv::Mat& _labels, std::vector<cv::Rect> &_bboxes);
        inline void posProcessBboxes(std::vector<cv::Rect> &_bboxes);
    };
}