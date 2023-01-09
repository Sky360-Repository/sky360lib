#include "connectedBlobDetection.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <execution>
#include <algorithm>

using namespace sky360lib::blobs;

ConnectedBlobDetection::ConnectedBlobDetection(size_t _numProcessesParallel)
    : m_sizeThreshold{DEFAULT_SIZE_THRESHOLD},
      m_areaThreshold{DEFAULT_AREA_THRESHOLD},
      m_numProcessesParallel{_numProcessesParallel},
      m_initialized{false}
{
    if (m_numProcessesParallel == DETECT_NUMBER_OF_THREADS)
    {
        m_numProcessesParallel = calcAvailableThreads();
    }
}

inline cv::KeyPoint convertFromRect(const cv::Rect &rect)
{
    static const float scale = 6.0f;
    const float size = (float)std::max(rect.width, rect.height) / scale;
    return cv::KeyPoint(rect.x + scale * size / 2.0f, rect.y + scale * size / 2.0f, size);
}

std::vector<cv::KeyPoint> ConnectedBlobDetection::detectKP(const cv::Mat &_image)
{
    std::vector<cv::Rect> bboxes;
    detect(_image, bboxes);
    std::vector<cv::KeyPoint> kps;
    std::transform(bboxes.begin(),
                   bboxes.end(),
                   std::back_inserter(kps),
                   [](const cv::Rect &r) -> cv::KeyPoint
                   { return convertFromRect(r); });
    return kps;
}

std::vector<cv::Rect> ConnectedBlobDetection::detectRet(const cv::Mat &_image)
{
    std::vector<cv::Rect> bboxes;
    detect(_image, bboxes);
    return bboxes;
}

// Joining bboxes together if they overlap
inline static void joinBBoxes(std::vector<cv::Rect> &_bboxes)
{
    bool bboxOverlap;
    do
    {
        bboxOverlap = false;
        for (size_t i{0}; i < _bboxes.size() - 1; ++i)
        {
            for (size_t j{i + 1}; j < _bboxes.size();)
            {
                if (sky360lib::rectsOverlap(_bboxes[i], _bboxes[j]))
                {
                    bboxOverlap = true;
                    const int xmax = std::max(_bboxes[i].x + _bboxes[i].width, _bboxes[j].x + _bboxes[j].width);
                    const int ymax = std::max(_bboxes[i].y + _bboxes[i].height, _bboxes[j].y + _bboxes[j].height);
                    _bboxes[i].x = std::min(_bboxes[i].x, _bboxes[j].x);
                    _bboxes[i].y = std::min(_bboxes[i].y, _bboxes[j].y);
                    _bboxes[i].width = xmax - _bboxes[i].x;
                    _bboxes[i].height = ymax - _bboxes[i].y;
                    _bboxes.erase(_bboxes.begin() + j);
                }
                else
                {
                    ++j;
                }
            }
        }
    } while (bboxOverlap);
}

inline static void applySizeCut(std::vector<cv::Rect> &_bboxes, const int _sizeThreshold, const int _areaThreshold)
{
    for (size_t i{0}; i < _bboxes.size();)
    {
        if ((_bboxes[i].width < _sizeThreshold) || (_bboxes[i].height < _sizeThreshold) || (_bboxes[i].area() < _areaThreshold))
        {
            _bboxes.erase(_bboxes.begin() + i);
        }
        else
        {
            ++i;
        }
    }
}

inline void ConnectedBlobDetection::posProcessBboxes(std::vector<cv::Rect> &_bboxes)
{
    const size_t numLabels = _bboxes.size();

    // Reseting returning bboxes to the MIN/MAX values
    for (size_t i{0}; i < numLabels; ++i)
    {
        _bboxes[i].x = _bboxes[i].y = INT_MAX;
        _bboxes[i].width = _bboxes[i].height = INT_MIN;
    }

    // Joining all parallel bboxes into one label
    for (size_t i{0}; i < m_numProcessesParallel; ++i)
    {
        const int addedY = (int)(m_imgSizesParallel[i]->originalPixelPos / m_imgSizesParallel[i]->width);
        const std::vector<cv::Rect> &bboxesParallel = m_bboxesParallel[i];
        for (size_t j{0}; j < numLabels; ++j)
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

    // Joining bboxes that are overlaping each other
    joinBBoxes(_bboxes);

    // Removing bboxes that are below threshold
    applySizeCut(_bboxes, m_sizeThreshold, m_areaThreshold);
}

// Finds the connected components in the image and returns a list of bounding boxes
bool ConnectedBlobDetection::detectOld(const cv::Mat &_image, std::vector<cv::Rect> &_bboxes)
{
    if (!m_initialized)
    {
        prepareParallel(_image);
        // Create a labels image to store the labels for each connected component
        m_initialized = true;
    }

    cv::Mat stats;
    cv::Mat centroids;

    // Use connected component analysis to find the blobs in the image, subtract 1 because the background is consured as label
    // CCL_SAUF        = 0,  //!< SAUF @cite Wu2009 algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity. The parallel implementation described in @cite Bolelli2017 is available for SAUF.
    // CCL_BBDT     = 1,  //!< BBDT @cite Grana2010 algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity. The parallel implementation described in @cite Bolelli2017 is available for both BBDT and SAUF.
    // CCL_SPAGHETTI   = 2,  //!< Spaghetti @cite Bolelli2019 algorithm for 8-way connectivity, Spaghetti4C @cite Bolelli2021 algorithm for 4-way connectivity. The parallel implementation described in @cite Bolelli2017 is available for both Spaghetti and Spaghetti4C.
    const int numLabels = cv::connectedComponentsWithStats(_image, m_labels, stats, centroids, 8, CV_32S, cv::CCL_SPAGHETTI) - 1;

    _bboxes.resize(numLabels);
    if (numLabels > 0)
    {
        for (int j{0}; j < numLabels; ++j)
        {
            const int label = j + 1;
            _bboxes[j].x = stats.at<int>(label, cv::CC_STAT_LEFT);
            _bboxes[j].y = stats.at<int>(label, cv::CC_STAT_TOP);
            _bboxes[j].width = stats.at<int>(label, cv::CC_STAT_WIDTH);
            _bboxes[j].height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        }

        // posProcessBboxes(_bboxes);

        return true;
    }

    return false;
}

// Finds the connected components in the image and returns a list of bounding boxes
bool ConnectedBlobDetection::detect(const cv::Mat &_image, std::vector<cv::Rect> &_bboxes)
{
    if (!m_initialized)
    {
        prepareParallel(_image);
        // Create a labels image to store the labels for each connected component
        m_initialized = true;
    }

    // Use connected component analysis to find the blobs in the image, subtract 1 because the background is consured as label
    // CCL_SAUF        = 0,  //!< SAUF @cite Wu2009 algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity. The parallel implementation described in @cite Bolelli2017 is available for SAUF.
    // CCL_BBDT     = 1,  //!< BBDT @cite Grana2010 algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity. The parallel implementation described in @cite Bolelli2017 is available for both BBDT and SAUF.
    // CCL_SPAGHETTI   = 2,  //!< Spaghetti @cite Bolelli2019 algorithm for 8-way connectivity, Spaghetti4C @cite Bolelli2021 algorithm for 4-way connectivity. The parallel implementation described in @cite Bolelli2017 is available for both Spaghetti and Spaghetti4C.
    const int numLabels = cv::connectedComponents(_image, m_labels, 8, CV_32S, cv::CCL_BOLELLI) - 1;

    _bboxes.resize(numLabels);
    if (numLabels > 0)
    {
        std::for_each(
            std::execution::par,
            m_processSeq.begin(),
            m_processSeq.end(),
            [&](int np)
            {
                // Reseting parallel bboxes to the MIN/MAX values
                m_bboxesParallel[np].resize(numLabels);
                for (int j{0}; j < numLabels; ++j)
                {
                    m_bboxesParallel[np][j].x = m_bboxesParallel[np][j].y = INT_MAX;
                    m_bboxesParallel[np][j].width = m_bboxesParallel[np][j].height = INT_MIN;
                }
                // Spliting the image into chuncks and processing
                const cv::Mat imgSplit(m_imgSizesParallel[np]->height, m_imgSizesParallel[np]->width, m_labels.type(),
                                       m_labels.data + (m_imgSizesParallel[np]->originalPixelPos * m_imgSizesParallel[np]->numBytesPerPixel));
                applyDetectBBoxes(imgSplit, m_bboxesParallel[np]);
            });

        posProcessBboxes(_bboxes);

        return true;
    }

    return false;
}

void ConnectedBlobDetection::applyDetectBBoxes(const cv::Mat &_labels, std::vector<cv::Rect> &_bboxes)
{
    int *pLabel = (int *)_labels.data;
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
