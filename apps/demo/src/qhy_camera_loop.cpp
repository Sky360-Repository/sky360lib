#include <iostream>
#include <string>
#include <algorithm>
#include <thread>

#include <easy/profiler.h>

#include "qhyCamera.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "profiling.hpp"

/////////////////////////////////////////////////////////////
// Default parameters
bool isVideoOpen = false;
cv::VideoWriter videoWriter;

/////////////////////////////////////////////////////////////
// Camera Detector
sky360lib::camera::QHYCamera qhyCamera;

/////////////////////////////////////////////////////////////
// Function Definitions
bool openQQYCamera();
inline bool getQhyCameraImage(cv::Mat &cameraFrame);
bool openVideo(const cv::Mat &frame);

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    EASY_PROFILER_ENABLE;

    if (!openQQYCamera())
    {
        return -1;
    }

    double exposure = (argc > 1 ? atoi(argv[1]) : 20000);
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);

    initFrequency();

    cv::namedWindow("Live Video", 0);

    cv::Mat frame, processedFrame, saveFrame;
    long numFrames{0};
    long totalNumFrames{0};
    double totalTime{0.0};
    double totalProcessedTime{0.0};
    double totalMeanProcessedTime{0.0};
    std::cout << "1" << std::endl;

    if (!getQhyCameraImage(frame))
    {
        std::cout << "Could not get image" << std::endl;
        return -1;
    }

    bool pause = false;
    std::cout << "Enter loop" << std::endl;
    while (true)
    {
        double startFrameTime = getAbsoluteTime();
        EASY_BLOCK("Loop pass");
        if (!pause)
        {
            double startProcessedTime = getAbsoluteTime();
            getQhyCameraImage(frame);
            double endProcessedTime = getAbsoluteTime();
            ++numFrames;
            totalProcessedTime += endProcessedTime - startProcessedTime;
            totalMeanProcessedTime += endProcessedTime - startProcessedTime;
            ++totalNumFrames;
            cv::imshow("Live Video", frame);
            cv::resizeWindow("Live Video", 1024, 1024);
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
                isVideoOpen = false;
                videoWriter.release();
            }
        }
        else if (key == '+')
        {
            exposure *= 1.1;
            std::cout << "Setting exposure to: " << exposure << std::endl;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
        }
        else if (key == '-')
        {
            exposure *= 0.9;
            std::cout << "Setting exposure to: " << exposure << std::endl;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
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
              << "Average Framerate: " << (totalNumFrames / totalMeanProcessedTime) << " fps" << std::endl;

    cv::destroyAllWindows();

    qhyCamera.close();

    profiler::dumpBlocksToFile("test_profile.prof");

    return 0;
}

bool openQQYCamera()
{
    auto cameras = qhyCamera.getCameras();
    if (cameras.size() == 0)
    {
        return false;
    }
    if (!qhyCamera.open(cameras.begin()->first))
    {
        std::cout << "Error opening camera" << std::endl;
        return false;
    }

    //qhyCamera.setStreamMode(sky360lib::camera::QHYCamera::SingleFrame);

    // check color camera
    if (qhyCamera.getCameraInfo()->isColor)
    {
        qhyCamera.debayer(false);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::RedWB, 70.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::GreenWB, 65.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::BlueWB, 88.0);
    }
    if (!qhyCamera.setControl(sky360lib::camera::QHYCamera::Gain, 30))
    {
        return false;
    }
    if (!qhyCamera.setControl(sky360lib::camera::QHYCamera::Offset, 0))
    {
        return false;
    }
    if (!qhyCamera.setControl(sky360lib::camera::QHYCamera::TransferBits, 16))
    {
        return false;
    }
    if (!qhyCamera.setBinMode(sky360lib::camera::QHYCamera::Bin_1x1))
    {
        return false;
    }
    return true;
}

inline bool getQhyCameraImage(cv::Mat &cameraFrame)
{
    return qhyCamera.getFrame(cameraFrame, true);
}

bool openVideo(const cv::Mat &frame)
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    auto name = "vo" + oss.str() + ".mkv";
    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    return videoWriter.open(name, codec, 10, frame.size(), true);
}