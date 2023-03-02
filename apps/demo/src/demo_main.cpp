#include <iostream>
#include <string>
#include <algorithm>
#include <thread>

#include <easy/profiler.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "bgs.hpp"
#include "profiling.hpp"
#include "connectedBlobDetection.hpp"

#include "demoUtils.hpp"
#include "demoVideoTracker.hpp"

/////////////////////////////////////////////////////////////
// Default parameters
int blur_radius{3};
bool applyGreyscale{true};
bool applyNoiseReduction{false};
int sensitivity{1};

/////////////////////////////////////////////////////////////
// Background subtractor to use
enum BGSType
{
    Vibe,
    WMV,
    WMVCL
};
std::unique_ptr<sky360lib::bgs::CoreBgs> bgsPtr{nullptr};

/////////////////////////////////////////////////////////////
// Blob Detector
sky360lib::blobs::ConnectedBlobDetection blobDetector;

/////////////////////////////////////////////////////////////
// Video Tracker
DemoVideoTracker videoTracker;

/////////////////////////////////////////////////////////////
// Function Definitions
std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type);
inline void appyPreProcess(const cv::Mat &input, cv::Mat &output);
inline void appyBGS(const cv::Mat &input, cv::Mat &output);
inline void applyTracker(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &frame);
inline void drawBboxes(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &frame);
inline void findBlobs(const cv::Mat &image, std::vector<cv::Rect> &blobs);
inline void drawBboxes(std::vector<cv::Rect> &keypoints, const cv::Mat &frame);
inline void outputBoundingBoxes(std::vector<cv::Rect> &bboxes);
int getIntArg(std::string arg);

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    EASY_PROFILER_ENABLE;

    std::string videoFile{"Dahua-20220901-184734.mp4"};
    // std::string videoFile{"birds_and_plane.mp4"};
    // std::string videoFile{"brad_drone_1.mp4"};

    // Setting some initial configurations
    cv::ocl::setUseOpenCL(true);
    if (cv::ocl::haveOpenCL())
    {
        std::cout << "Has OpenCL support, using: " << (cv::ocl::useOpenCL() ? "Yes" : "No") << std::endl;
    }

    initFrequency();

    std::cout << "Available number of concurrent threads = " << std::thread::hardware_concurrency() << std::endl;

    bgsPtr = createBGS(BGSType::WMV);
    cv::VideoCapture cap;

    if (argc > 1)
    {
        int camNum = getIntArg(argv[1]);
        if (camNum >= 0)
        {
            cap.open(camNum);
        }
        else
        {
            cap.open(argv[1]);
        }
    }
    else
    {
        cap.open(videoFile);
    }

    // int camNum = std::stoi(argv[1]);
    // cap.open(camNum);
    if (!cap.isOpened())
    {
        std::cout << "***Could not initialize capturing...***" << std::endl;
        return -1;
    }

    double frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Capture size: " << (int)frameWidth << " x " << (int)frameHeight << std::endl;

    cv::namedWindow("BGS Demo", 0);
    cv::namedWindow("Live Video", 0);

    cv::Mat frame, frame16, processedFrame;
    long numFrames{0};
    long totalNumFrames{0};
    double totalTime{0.0};
    double totalProcessedTime{0.0};
    double totalMeanProcessedTime{0.0};

    cap.read(frame);
    if (frame.type() != CV_8UC3)
    {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }
    cv::Mat bgsMask{frame.size(), CV_8UC1};

    std::vector<cv::Rect> bboxes;
    bool pause = false;
    std::cout << "Enter loop" << std::endl;
    while (true)
    {
        double startFrameTime = getAbsoluteTime();
        EASY_BLOCK("Loop pass");
        if (!pause)
        {
            double startProcessedTime = getAbsoluteTime();
            EASY_BLOCK("Capture");
            cap.read(frame);
            if (frame.empty())
            {
                std::cout << "No image" << std::endl;
                break;
            }
            frame.convertTo(frame16, CV_16UC3, 256.0f);
            EASY_END_BLOCK;
            EASY_BLOCK("Process");
            appyPreProcess(frame16, processedFrame);
            appyBGS(processedFrame, bgsMask);
            findBlobs(bgsMask, bboxes);
            //applyTracker(blobs, processedFrame);
            double endProcessedTime = getAbsoluteTime();
            EASY_END_BLOCK;
            EASY_BLOCK("Drawing bboxes");
            drawBboxes(bboxes, bgsMask);
            drawBboxes(bboxes, frame);
            EASY_END_BLOCK;
            ++numFrames;
            totalProcessedTime += endProcessedTime - startProcessedTime;
            totalMeanProcessedTime += endProcessedTime - startProcessedTime;
            ++totalNumFrames;
            EASY_BLOCK("Show/resize windows");
            cv::imshow("BGS Demo", bgsMask);
            cv::resizeWindow("BGS Demo", 1024, 1024);
            cv::imshow("Live Video", frame);
            cv::resizeWindow("Live Video", 1024, 1024);
            EASY_END_BLOCK;
        }
        char key = (char)cv::waitKey(1);
        if (key == 27)
        {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
        else if (key == 32)
        {
            pause = !pause;
            outputBoundingBoxes(bboxes);
        }
        else if (key == '+')
        {
            auto params = (sky360lib::bgs::WMVParams&)(bgsPtr->getParameters());
            float threshold = params.getThreshold();
            std::cout << "Got threshold: " << threshold << std::endl;
            params.setThreshold(threshold + 5);
        }
        else if (key == '-')
        {
            auto params = (sky360lib::bgs::WMVParams&)(bgsPtr->getParameters());
            float threshold = params.getThreshold();
            std::cout << "Got threshold: " << threshold << std::endl;
            params.setThreshold(threshold - 5);
        }
        double endFrameTime = getAbsoluteTime();
        totalTime += endFrameTime - startFrameTime;
        if (totalTime > 2.0)
        {
            std::cout << "Framerate: " << (numFrames / totalProcessedTime) << " fps" << std::endl;
            totalTime = 0.0;
            totalProcessedTime = 0.0;
            numFrames = 0;
        }
        EASY_END_BLOCK;
    }
    std::cout << "Exit loop\n"
              << std::endl;
    std::cout << std::endl
              << "Mean Framerate: " << (totalNumFrames / totalMeanProcessedTime) << " fps" << std::endl;

    cap.release();

    cv::destroyAllWindows();

    profiler::dumpBlocksToFile("test_profile.prof");

    return 0;
}

std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type)
{
    switch (_type)
    {
    case BGSType::Vibe:
        return std::make_unique<sky360lib::bgs::Vibe>(sky360lib::bgs::VibeParams(50, 24, 1, 2));
    case BGSType::WMV:
        return std::make_unique<sky360lib::bgs::WeightedMovingVariance>();
    case BGSType::WMVCL:
        return std::make_unique<sky360lib::bgs::WeightedMovingVarianceCL>();
    default:
        return std::make_unique<sky360lib::bgs::WeightedMovingVariance>();
    }
}

// Do image pre-processing
inline void appyPreProcess(const cv::Mat &input, cv::Mat &output)
{
    EASY_FUNCTION(profiler::colors::Green);
    cv::Mat tmpFrame;

    EASY_BLOCK("Greyscale");
    if (applyGreyscale)
        cv::cvtColor(input, tmpFrame, cv::COLOR_RGB2GRAY);
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
inline void appyBGS(const cv::Mat &input, cv::Mat &output)
{
    EASY_FUNCTION(profiler::colors::Red);
    bgsPtr->apply(input, output);
}

inline void applyTracker(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &frame)
{
    EASY_FUNCTION(profiler::colors::Yellow);
    videoTracker.create_trackers_from_keypoints(keypoints, frame);
}

inline void outputBoundingBoxes(std::vector<cv::Rect> &bboxes)
{
    std::cout << "Bounding boxes" << std::endl;
    for (auto bb : bboxes)
    {
        std::cout << bb << std::endl;
    }
}

inline void drawBboxes(std::vector<cv::Rect> &bboxes, const cv::Mat &frame)
{
    for (auto bb : bboxes)
    {
        cv::rectangle(frame, bb, cv::Scalar(255, 0, 255), 2);
    }
}

// Finds the connected components in the image and returns a list of bounding boxes
inline void findBlobs(const cv::Mat &image, std::vector<cv::Rect> &blobs)
{
    EASY_FUNCTION(profiler::colors::Blue);

    blobDetector.detect(image, blobs);
}

int getIntArg(std::string arg)
{
    std::size_t pos{};
    try
    {
        const int argNum{std::stoi(arg, &pos)};
        return pos == arg.size() ? argNum : -1;
    }
    catch (std::exception const &ex)
    {
        return -1;
    }
}