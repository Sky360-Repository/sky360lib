#pragma once

#include "coreUtils.hpp"

#include <opencv2/core.hpp>

namespace sky360lib::blobs
{
    struct ConnectedBlobDetectionParams final
    {
        static const int DEFAULT_SIZE_THRESHOLD = 5;
        static const int DEFAULT_AREA_THRESHOLD = 25;
        static const int DEFAULT_MIN_DISTANCE = 5;

        ConnectedBlobDetectionParams()
            : ConnectedBlobDetectionParams(DEFAULT_SIZE_THRESHOLD, DEFAULT_AREA_THRESHOLD, DEFAULT_MIN_DISTANCE)
        {
        }

        ConnectedBlobDetectionParams(int _sizeThreshold, int _areaThreshold, int _minDistance)
            : sizeThreshold{_sizeThreshold}, areaThreshold{_areaThreshold}, minDistance{_minDistance}, minDistanceSquared{_minDistance * _minDistance}
        {
        }

        inline void setSizeThreshold(int _threshold) { sizeThreshold = std::max(_threshold, 2); }
        inline void setAreaThreshold(int _threshold) { areaThreshold = std::max(_threshold, sizeThreshold * sizeThreshold); }
        inline void setMinDistance(int _minDistance)
        {
            minDistance = std::max(_minDistance, 2);
            minDistanceSquared = minDistance * minDistance;
        }

        int sizeThreshold;
        int areaThreshold;
        int minDistance;
        int minDistanceSquared;
    };

    class ConnectedBlobDetection final
    {
    public:
        /// Detects the number of available threads to use
        /// Will set the number fo threads to the number of avaible threads - 1
        static const size_t DETECT_NUMBER_OF_THREADS{0};

        ConnectedBlobDetection(const ConnectedBlobDetectionParams &_params = ConnectedBlobDetectionParams(),
                               size_t _numProcessesParallel = DETECT_NUMBER_OF_THREADS);

        // Finds the connected components in the image and returns a list of bounding boxes
        bool detect(const cv::Mat &_image, std::vector<cv::Rect> &_bboxes);

        inline void setSizeThreshold(int _threshold) { m_params.setSizeThreshold(_threshold); }
        inline void setAreaThreshold(int _threshold) { m_params.setSizeThreshold(_threshold); }
        inline void setMinDistance(int _distance) { m_params.setMinDistance(_distance); }

        // Finds the connected components in the image and returns a list of keypoints
        // This function uses detect and converts from Rect to KeyPoints using a fixed scale
        std::vector<cv::KeyPoint> detectKP(const cv::Mat &_image);

        // Finds the connected components in the image and returns a list of bounding boxes
        std::vector<cv::Rect> detectRet(const cv::Mat &_image);

    private:
        ConnectedBlobDetectionParams m_params;
        size_t m_numProcessesParallel;
        bool m_initialized;
        cv::Mat m_labels;
        std::vector<size_t> m_processSeq;
        std::vector<std::unique_ptr<ImgSize>> m_imgSizesParallel;
        std::vector<std::vector<cv::Rect>> m_bboxesParallel;

        void prepareParallel(const cv::Mat &_image);
        static void applyDetectBBoxes(const cv::Mat &_labels, std::vector<cv::Rect> &_bboxes);
        inline void posProcessBboxes(std::vector<cv::Rect> &_bboxes);
    };
}