#include <iostream>
#include <string>

#include <easy/profiler.h>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/tracking.hpp>

#include "bgs.hpp"
#include "tracker.hpp"
#include "profiling.hpp"

using namespace sky360lib::bgs;
using namespace sky360lib::tracking;

// Default parameters
int blur_radius{3};
bool applyGreyscale{true};
bool applyNoiseReduction{true};

// Background subtractor to use
WeightedMovingVariance bgs;
//WeightedMovingVarianceHalide bgs;
// Vibe bgs;

// Tracker to use
// cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create(); //OpenCV Tracker
cv::Ptr<TrackerCSRT> tracker = TrackerCSRT::create(); // sky360lib Tracker implementation

// Do image pre-processing
inline void appyPreProcess(const cv::Mat& input, cv::Mat& output)
{
    EASY_FUNCTION(profiler::colors::Green);
    cv::Mat tmpFrame;

    EASY_BLOCK("Greyscale");
    if (applyGreyscale)
        cv::cvtColor(input, tmpFrame, cv::COLOR_BGR2GRAY);
    else
        tmpFrame = input;
    EASY_END_BLOCK;
    EASY_BLOCK("Noise Reduction");
    if (applyNoiseReduction)
        cv::GaussianBlur(tmpFrame, output, cv::Size(blur_radius, blur_radius), 0);
    else
        output = tmpFrame;
    EASY_END_BLOCK;
}

// Apply background subtraction
inline void appyBGS(const cv::Mat& input, cv::Mat& output)
{
    EASY_FUNCTION(profiler::colors::Red);
    bgs.apply(input, output);
}

// Main entry point
int main(int argc, const char **argv)
{
    EASY_PROFILER_ENABLE;

    cv::VideoCapture cap;

    // if (argc < 2) {
    //     std::cout << "Need one parameter as camera number" << std::endl;
    //     return -1;
    // }

    initFrequency();

    // int camNum = std::stoi(argv[1]);
    // cap.open(camNum);
    cap.open("Dahua-20220901-184734.mp4");
    if (!cap.isOpened())
    {
        std::cout << "***Could not initialize capturing...***\n";
        return -1;
    }

    double frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Capture size: " << (int)frameWidth << " x " << (int)frameHeight << std::endl;

    cv::namedWindow("BGS Demo", 0);
    cv::namedWindow("Live Video", 0);

    cv::Mat frame, processedFrame;
    long numFrames = 0;
    double totalTime = 0;

    cap.read(frame);
    if (frame.type() != CV_8UC3)
    {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    cv::Mat bgsMask{frame.size(), CV_8UC1};

    // Applying first time for initialization of algo
    appyPreProcess(frame, processedFrame);
    appyBGS(processedFrame, bgsMask);

    //cv::Rect boundingBox = cv::Rect(0, 0, frame.size().width, frame.size().height);
    //tracker->init(bgsMask, boundingBox);

    cv::imshow("BGS Demo", frame);

    std::cout << "Enter loop" << std::endl;
    while (true)
    {
        cap.read(frame);
        if (frame.empty())
        {
            std::cout << "No image" << std::endl;
            break;
        }
        double startTime = getAbsoluteTime();
        EASY_BLOCK("Doing process");
        appyPreProcess(frame, processedFrame);
        appyBGS(processedFrame, bgsMask);
        EASY_END_BLOCK;
        // if (tracker->update(bgsMask, boundingBox))
        // {
        //     cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
        // }
        double endTime = getAbsoluteTime();
        totalTime += endTime - startTime;
        ++numFrames;

        if (numFrames % 100 == 0)
        {
            std::cout << "Framerate: " << (numFrames / totalTime) << " fps" << std::endl;
            totalTime = 0;
            numFrames = 0;
        }
        cv::imshow("BGS Demo", bgsMask);
        cv::resizeWindow("BGS Demo", 1024, 1024);
        cv::imshow("Live Video", frame);
        cv::resizeWindow("Live Video", 1024, 1024);

        if ((char)cv::waitKey(1) == 27)
        {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
    }
    std::cout << "Exit loop\n"
              << std::endl;

    cap.release();

    cv::destroyAllWindows();

    profiler::dumpBlocksToFile("test_profile.prof");

    return 0;
}
