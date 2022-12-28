#pragma once

#include <opencv2/videoio.hpp>

inline double bbox_overlap(const cv::Rect& bbox1, const cv::Rect& bbox2)
{
    // determine the coordinates of the intersection rectangle
    const int x_left = std::max(bbox1.x, bbox2.x);
    const int y_top = std::max(bbox1.y, bbox2.y);
    const int x_right = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    const int y_bottom = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

    if (x_right < x_left || y_bottom < y_top)
        return 0.0;

    // The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box.
    // NOTE: We MUST ALWAYS add +1 to calculate area when working in screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    // is the bottom right pixel. If we DON'T add +1, the result is wrong.
    const int intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1);

    // compute the area of both AABBs
    const int bb1_area = (bbox1.width + 1) * (bbox1.height + 1);
    const int bb2_area = (bbox2.width + 1) * (bbox2.height + 1);

    // compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    return (double)intersection_area / (double)(bb1_area + bb2_area - intersection_area);
}

// Utility function to determine if a bounding box 1 contains bounding box 2. In order to make tracking more efficient
// we try not to track sections of the same point of interest (blob)
inline bool bbox1_contain_bbox2(const cv::Rect& bbox1, const cv::Rect& bbox2)
{
    return (bbox2.x > bbox1.x) && 
        (bbox2.y > bbox1.y) && 
        ((bbox2.x + bbox2.width) < (bbox1.x + bbox1.width)) && 
        ((bbox2.y + bbox2.height) < (bbox1.y + bbox1.height));
}

// Utility function to convert key points in to a bounding box
// The bounding box is used for track validation (if enabled) and will be displayed by the visualiser
// as it tracks a point of interest (blob) on the frame
inline cv::Rect kp_to_bbox(const cv::KeyPoint &kp)
{
    static const float scale = 20.0f;
    return cv::Rect((int)(kp.pt.x - (scale * kp.size * 0.5f)),
                    (int)(kp.pt.y - (scale * kp.size * 0.5f)),
                    (int)(scale * kp.size),
                    (int)(scale * kp.size));
}

inline size_t calc_centre_point_distance(const cv::Rect& bbox1, const cv::Rect& bbox2)
{
    // euclidean = math.sqrt((x2-x1)**2+(y2-y1)**2)
    size_t cx = (bbox2.x + (bbox2.width / 2)) - (bbox1.x + (bbox1.width / 2));
    size_t cy = (bbox2.y + (bbox2.height / 2)) - (bbox1.y + (bbox1.height / 2));
    return std::sqrt(cx * cx + cy * cy);
}

inline bool are_bboxes_equal(const cv::Rect &b1, const cv::Rect &b2)
{
    return (b1.x == b2.x && b1.y == b2.y && b1.width == b2.width && b1.height == b2.height);
}

