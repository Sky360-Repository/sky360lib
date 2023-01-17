#include <iostream>
#include <string>
#include <algorithm>
#include <thread>

#include <easy/profiler.h>

#include "qhyCamera.hpp"

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
bool applyNoiseReduction{true};
int sensitivity{1};

/////////////////////////////////////////////////////////////
// Background subtractor to use
enum BGSType
{
    Vibe
    ,WMV
    ,WMVCL
    //,WMVHalide
};
std::unique_ptr<sky360lib::bgs::CoreBgs> bgsPtr{nullptr};

/////////////////////////////////////////////////////////////
// Blob Detector
sky360lib::blobs::ConnectedBlobDetection blobDetector;

/////////////////////////////////////////////////////////////
// Video Tracker
DemoVideoTracker videoTracker;

sky360lib::camera::QHYCamera qhyCamera;
cv::VideoWriter videoWriter;
bool isVideoOpen = false;

/////////////////////////////////////////////////////////////
// Function Definitions
std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type);
inline void appyPreProcess(const cv::Mat &input, cv::Mat &output);
inline void appyBGS(const cv::Mat &input, cv::Mat &output);
inline void applyTracker(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &frame);
inline void drawBboxes(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &frame);
inline std::vector<cv::Rect> findBlobs(const cv::Mat &image);
inline void drawBboxes(std::vector<cv::Rect> &keypoints, const cv::Mat &frame);
inline void outputBoundingBoxes(std::vector<cv::Rect> &bboxes);

bool openQQYCamera();
inline void getQhyCameraImage(cv::Mat & cameraFrame);

bool openVideo(const cv::Mat& frame)
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    auto name = "vo" + oss.str() + ".mkv";
    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    return videoWriter.open(name , codec, 10, frame.size(), true);
}

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    cv::namedWindow("QHY", 0);

    if (!openQQYCamera())
    {
        return -1;
    }

    double exposure = (argc > 1 ? atoi(argv[1]): 20000);
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);

    // cv::Mat cameraFrame;
    // while (true)
    // {
    //     getQhyCameraImage(cameraFrame);
    //     cv::imshow("QHY", cameraFrame);
    //     cv::resizeWindow("QHY", 1024, 1024);

    //     char key = (char)cv::waitKey(1);
    //     if (key == 27)
    //     {
    //         std::cout << "Escape key pressed" << std::endl;
    //         break;
    //     } 
    //     else if (key == '+')
    //     {
    //         exposure += 1000;
    //         std::cout << "Setting exposure to: " << exposure << std::endl;
    //         qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
    //     }
    //     else if (key == '-')
    //     {
    //         exposure -= 1000;
    //         std::cout << "Setting exposure to: " << exposure << std::endl;
    //         qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
    //     }
    // }
    // qhyCamera.close();
    // return 0;

    const auto concurrentThreads = std::thread::hardware_concurrency();
    std::cout << "Available number of concurrent threads = " << concurrentThreads << std::endl;

    EASY_PROFILER_ENABLE;

    bgsPtr = createBGS(BGSType::WMV);

    cv::VideoCapture cap;

    // cv::setUseOpenVX(true);
    cv::ocl::setUseOpenCL(true);
    if (cv::ocl::haveOpenCL())
        std::cout << "Has OpenCL support, using it on OpenCV" << std::endl;

    initFrequency();

    // int camNum = std::stoi(argv[1]);
    // cap.open(camNum);
    // cap.open("Dahua-20220901-184734.mp4");
    // if (!cap.isOpened())
    // {
    //     std::cout << "***Could not initialize capturing...***" << std::endl;
    //     return -1;
    // }

    // double frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    // double frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // std::cout << "Capture size: " << (int)frameWidth << " x " << (int)frameHeight << std::endl;

    cv::namedWindow("BGS Demo", 0);
    cv::namedWindow("Live Video", 0);

    cv::Mat frame, processedFrame;
    long numFrames{0};
    long totalNumFrames{0};
    double totalTime{0.0};
    double totalProcessedTime{0.0};
    double totalMeanProcessedTime{0.0};

    getQhyCameraImage(frame);
    // cap.read(frame);
    if (frame.type() != CV_8UC3)
    {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    cv::Mat bgsMask{frame.size(), CV_8UC1};

    // Applying first time for initialization of algo
    appyPreProcess(frame, processedFrame);
    appyBGS(processedFrame, bgsMask);
    // detector = createBlobDetector(bgsMask);

    cv::imshow("BGS Demo", frame);

    std::vector<cv::Rect> bboxes;
    bool pause = false;
    bool doBlobDetection = false;
    std::cout << "Enter loop" << std::endl;
    while (true)
    {
        double startFrameTime = getAbsoluteTime();
        EASY_BLOCK("Loop pass");
        if (!pause)
        {
            EASY_BLOCK("Capture");
            double startProcessedTime = getAbsoluteTime();
            //cap.read(frame);
            getQhyCameraImage(frame);
            if (frame.empty())
            {
                std::cout << "No image" << std::endl;
                break;
            }
            EASY_END_BLOCK;
            EASY_BLOCK("Process");
            appyPreProcess(frame, processedFrame);
            appyBGS(processedFrame, bgsMask);
            // applyTracker(blobs, processedFrame);
            if (doBlobDetection)
                bboxes = findBlobs(bgsMask);
            double endProcessedTime = getAbsoluteTime();
            EASY_END_BLOCK;
            EASY_BLOCK("Drawing bboxes");
            if (doBlobDetection)
            {
                drawBboxes(bboxes, bgsMask);
                drawBboxes(bboxes, frame);
            }
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
            if (isVideoOpen)
            {
                videoWriter.write(frame);
            }
        }
        char key = (char)cv::waitKey(1);
        if (key == 27)
        {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
        else if (key == 32)
        {
            std::cout << "Pausing" << std::endl;
            pause = !pause;
            outputBoundingBoxes(bboxes);
        } 
        else if (key == 'v')
        {
            if (!isVideoOpen)
            {
                std::cout << "Start recording" << std::endl;
                isVideoOpen = openVideo(frame);
            }
            else
            {
                std::cout << "End recording" << std::endl;
                videoWriter.release();
            }
        }
        else if (key == 'b')
        {
            doBlobDetection = !doBlobDetection;
            std::cout << "Blob Detection: " << doBlobDetection << std::endl;
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

    qhyCamera.close();

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
    // case BGSType::WMVHalide:
    //     return std::make_unique<sky360lib::bgs::WeightedMovingVarianceHalide>();
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
inline std::vector<cv::Rect> findBlobs(const cv::Mat &image)
{
    EASY_FUNCTION(profiler::colors::Blue);

    std::vector<cv::Rect> blobs;
    blobDetector.detect(image, blobs);

    return blobs;
}

bool openQQYCamera()
{
    auto cameras = qhyCamera.getCameras();
    if (cameras.size() == 0)
    {
        return false;
    }
    if (!qhyCamera.open(cameras[0].id))
    {
        std::cout << "Error opening camera" << std::endl;
        return false;
    }

    // check color camera
    if (cameras[0].isColor)
    {
        qhyCamera.debayer(false);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::RedWB, 76.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::GreenWB, 58.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::BlueWB, 64.0);
    }
    if (!qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Gain, 30))
    {
        return false;
    }
    if (!qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Offset, 0))
    {
        return false;
    }
    if (!qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::TransferBits, 16))
    {
        return false;
    }
    if (!qhyCamera.setBinMode(1, 1))
    {
        return false;
    }
    return true;
}

inline void getQhyCameraImage(cv::Mat & cameraFrame)
{
    auto qhyframe = qhyCamera.getFrame();
    if (qhyframe == nullptr)
    {
        return;
    }
    //const cv::Mat imgQHY(2048, 3056, CV_8UC3, (int8_t*)qhyframe);
    const cv::Mat imgQHY(2048, 3056, CV_16UC1, (int8_t*)qhyframe);
    //const cv::Mat imgQHY(1024, 1528, CV_16UC1, (int8_t*)qhyframe);
    cv::cvtColor(imgQHY, cameraFrame, cv::COLOR_BayerGR2BGR);
    cameraFrame.convertTo(cameraFrame, CV_8U, 1/256.0f);
    //imgQHY.convertTo(cameraFrame, CV_8U, 1/256.0f);
}
