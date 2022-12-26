#include "connectedBlobDetection.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <execution>
#include <algorithm>

using namespace sky360lib::blobs;

ConnectedBlobDetection::ConnectedBlobDetection(size_t _numProcessesParallel)
    : m_numProcessesParallel{_numProcessesParallel}, 
    m_initialized{false}
{
    if (m_numProcessesParallel == DETECT_NUMBER_OF_THREADS)
    {
        m_numProcessesParallel = calcAvailableThreads();
    }
}

// Finds the connected components in the image and returns a list of bounding boxes
bool ConnectedBlobDetection::detect(const cv::Mat &_image, std::vector<cv::Rect> &_bboxes)
{
    if (!m_initialized)
    {
        prepareParallel(_image);
        // Create a labels image to store the labels for each connected component
        m_labels.create(_image.size(), CV_32SC1);
        m_initialized = true;
    }

    // Use connected component analysis to find the blobs in the image, subtract 1 because the background is consured as label
    const int numLabels = cv::connectedComponents(_image, m_labels, 8) - 1;

    _bboxes.resize(numLabels);
    if (numLabels > 0)
    {
        // Reseting returning bboxes to the MIN/MAX values
        for (int i{0}; i < numLabels; ++i)
        {
            _bboxes[i].x = _bboxes[i].y = INT_MAX;
            _bboxes[i].width = _bboxes[i].height = INT_MIN;
        }

        // Reseting parallel bboxes to the MIN/MAX values
        for (size_t i{0}; i < m_numProcessesParallel; ++i)
        {
            m_bboxesParallel[i].resize(numLabels);
            for (int j{0}; j < numLabels; ++j)
            {
                m_bboxesParallel[i][j].x = m_bboxesParallel[i][j].y = INT_MAX;
                m_bboxesParallel[i][j].width = m_bboxesParallel[i][j].height = INT_MIN;
            }
        }

        std::for_each(
            std::execution::par,
            m_processSeq.begin(),
            m_processSeq.end(),
            [&](int np)
            {
                const cv::Mat imgSplit(m_imgSizesParallel[np]->height, m_imgSizesParallel[np]->width, m_labels.type(),
                                        m_labels.data + (m_imgSizesParallel[np]->originalPixelPos * m_imgSizesParallel[np]->numBytesPerPixel));
                applyDetect(imgSplit, m_bboxesParallel[np]);
            });

        //size_t i{1};
        for (size_t i{0}; i < m_numProcessesParallel; ++i)
        {
            const int addedY = (int)(m_imgSizesParallel[i]->originalPixelPos / m_imgSizesParallel[i]->width);
            const std::vector<cv::Rect> &bboxesParallel = m_bboxesParallel[i];
            for (int j{0}; j < numLabels; ++j)
            {
                // If the coordinates for the label were altered, process
                if (bboxesParallel[j].x != INT_MAX)
                {
                    _bboxes[j].x = std::min(_bboxes[j].x, bboxesParallel[j].x);
                    _bboxes[j].y = std::min(_bboxes[j].y, bboxesParallel[j].y + addedY);
                    _bboxes[j].width = std::max(_bboxes[j].width, (bboxesParallel[j].width - _bboxes[j].x) + 1);
                    _bboxes[j].height = std::max(_bboxes[j].height, ((bboxesParallel[j].height + addedY) - _bboxes[j].y) + 1);
                }
            }
        }

        return true;
    }

    return false;
}

void ConnectedBlobDetection::applyDetect(const cv::Mat &_labels, std::vector<cv::Rect> &_bboxes)
{
    int* pLabel = (int*)_labels.data;
    for (int r = 0; r < _labels.rows; r++)
    {
        for (int c = 0; c < _labels.cols; c++)
        {
            const int label = *pLabel - 1;
            if (label >= 0)
            {
                _bboxes[label].x = std::min(_bboxes[label].x, c);
                _bboxes[label].y = std::min(_bboxes[label].y, r);
                _bboxes[label].width = std::max(_bboxes[label].width, c);
                _bboxes[label].height = std::max(_bboxes[label].height, r);
            }
            ++pLabel;
        }
    }
}

void ConnectedBlobDetection::prepareParallel(const cv::Mat &_image)
{
    m_imgSizesParallel.resize(m_numProcessesParallel);
    m_processSeq.resize(m_numProcessesParallel);
    m_bboxesParallel.resize(m_numProcessesParallel);
    size_t y{0};
    size_t h{_image.size().height / m_numProcessesParallel};
    for (size_t i{0}; i < m_numProcessesParallel; ++i)
    {
        m_processSeq[i] = i;
        if (i == (m_numProcessesParallel - 1))
        {
            h = _image.size().height - y;
        }
        m_imgSizesParallel[i] = ImgSize::create(_image.size().width, h,
                                                4,
                                                y * _image.size().width);
        y += h;
    }
}
