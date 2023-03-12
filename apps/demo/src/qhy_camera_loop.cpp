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

#include "profiling.hpp"

/////////////////////////////////////////////////////////////
// Default parameters
const int DEFAULT_WINDOW_WIDTH{1024};
const int DEFAULT_BOX_SIZE{500};

bool isVideoOpen = false;
bool isBoxSelected = false;
int frameWidth;
int frameHeight;
cv::VideoWriter videoWriter;

cv::Rect fullFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Rect tempFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};

/////////////////////////////////////////////////////////////
// Camera Detector
sky360lib::camera::QHYCamera qhyCamera;

/////////////////////////////////////////////////////////////
// Function Definitions
inline void drawBoxes(const cv::Mat &frame);
void writeText(const cv::Mat _frame, std::string _text, int _line);
bool openQQYCamera();
inline bool getQhyCameraImage(cv::Mat &cameraFrame);
bool openVideo(const cv::Mat &frame, double meanFps);
inline void debayerImage(const cv::Mat &imageIn, cv::Mat &imageOut);

static void changeParam(int value, void *paramP)
{
    long param = (long)paramP;
    switch (param)
    {
    case sky360lib::camera::QHYCamera::ControlParam::Gain:
        qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Gain, (double)value);
        break;
    }
}

void MouseCallBackFunc(int event, int x, int y, int, void *)
{
    if (event == cv::EVENT_LBUTTONUP)
    {
        fullFrameBox = tempFrameBox;
        if (!isBoxSelected)
        {
            cv::namedWindow("Window Cut", cv::WINDOW_AUTOSIZE);
            isBoxSelected = true;
        }
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        //tempFrameBox
        if (x > (frameWidth - (tempFrameBox.width / 2.0)))
        {
            x = frameWidth - (tempFrameBox.width / 2.0);
        }
        else if (x < (tempFrameBox.width / 2.0))
        {
            x = (tempFrameBox.width / 2.0);
        }
        if (y > (frameHeight - (tempFrameBox.height / 2.0)))
        {
            y = frameHeight - (tempFrameBox.height / 2.0);
        }
        else if (y < (tempFrameBox.height / 2.0))
        {
            y = (tempFrameBox.height / 2.0);
        }
        tempFrameBox.x = x - (tempFrameBox.width / 2.0);
        tempFrameBox.y = y - (tempFrameBox.height / 2.0);
    }
}

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    EASY_PROFILER_ENABLE;

    if (!openQQYCamera())
    {
        return -1;
    }
    std::cout << qhyCamera.getCameraInfo()->toString() << std::endl;

    double aspectRatio{(double)qhyCamera.getCameraInfo()->maxImageWidth / (double)qhyCamera.getCameraInfo()->maxImageHeight};

    double exposure = (argc > 1 ? atoi(argv[1]) : 20000);
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);

    const auto concurrentThreads = std::thread::hardware_concurrency();
    std::cout << "Available number of concurrent threads = " << concurrentThreads << std::endl;

    if (cv::ocl::haveOpenCL())
    {
        std::cout << "Has OpenCL support, using it on OpenCV" << std::endl;
    }

    initFrequency();

    cv::namedWindow("Live Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live Video", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_WIDTH / aspectRatio);
    cv::setMouseCallback("Live Video", MouseCallBackFunc, NULL);

    int gain = 30;
    cv::createTrackbar("Gain:", "Live Video", nullptr, 50, changeParam, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::Gain);
    cv::setTrackbarPos("Gain:", "Live Video", gain);

    cv::Mat frame, processedFrame, saveFrame, frameDebayered;
    long numFrames{0};
    long totalNumFrames{0};
    double totalTime{0.0};
    double totalProcessedTime{0.0};
    double totalMeanProcessedTime{0.0};
    double lastProcessingFPS{0.0};
    double cameraTime{0.0};
    double cameraFPS{0.0};

    getQhyCameraImage(frame);

    frameWidth = frame.size().width;
    frameHeight = frame.size().height;

    cv::Mat videoFrame{frame.size(), CV_8UC3};

    std::vector<cv::Rect> bboxes;
    bool run = true;
    bool pause = false;
    bool doBlobDetection = false;
    std::cout << "Enter loop" << std::endl;
    while (run)
    {
        double startFrameTime = getAbsoluteTime();
        if (!pause)
        {
            double startProcessedTime = getAbsoluteTime();
            getQhyCameraImage(frame);
            cameraTime += qhyCamera.getLastFrameCaptureTime();
            double endProcessedTime = getAbsoluteTime();
            debayerImage(frame, frameDebayered);

            if (isBoxSelected)
            {
                cv::Mat cropFrame = frameDebayered(fullFrameBox);
                cv::imshow("Window Cut", cropFrame);
            }

            drawBoxes(frameDebayered);
            if (frameDebayered.elemSize1() > 1)
            {
                frameDebayered.convertTo(videoFrame, CV_8U, 1 / 256.0f);
            }
            else
            {
                videoFrame = frameDebayered;
            }
            writeText(videoFrame, "Exposure: " + std::to_string(exposure / 1000.0) + " ms ('+' to +10%, '-' to -10%)", 1);
            writeText(videoFrame, "Bits: " + std::to_string(qhyCamera.getCameraParams().transferBits) + " ('1' to 8 bits, '2' to 16 bits)", 2);
            writeText(videoFrame, "Blob Detection: " + std::string(doBlobDetection ? "On" : "Off") + " ('b' to toggle)", 4);
            writeText(videoFrame, "Video Recording: " + std::string(isVideoOpen ? "Yes" : "No") + " ('v' to toggle)", 5);
            writeText(videoFrame, "Capture: " + std::to_string(cameraFPS) + " fps", 7);

            ++numFrames;
            totalProcessedTime += endProcessedTime - startProcessedTime;
            totalMeanProcessedTime += endProcessedTime - startProcessedTime;
            ++totalNumFrames;
            cv::imshow("Live Video", videoFrame);
            if (isVideoOpen)
            {
                videoWriter.write(videoFrame);
            }
        }

        char key = (char)cv::waitKey(1);
        switch (key)
        {
        case 27:
            std::cout << "Escape key pressed" << std::endl;
            run = false;
            break;
        case ' ':
            std::cout << "Pausing" << std::endl;
            pause = !pause;
            break;
        case 'v':
            if (!isVideoOpen)
            {
                std::cout << "Start recording" << std::endl;
                isVideoOpen = openVideo(frame, totalNumFrames / totalMeanProcessedTime);
            }
            else
            {
                std::cout << "End recording" << std::endl;
                isVideoOpen = false;
                videoWriter.release();
            }
            break;
        case '+':
            exposure *= 1.1;
            std::cout << "Setting exposure to: " << exposure << std::endl;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
            break;
        case '-':
            exposure *= 0.9;
            std::cout << "Setting exposure to: " << exposure << std::endl;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
            break;
        case '1':
            std::cout << "Setting bits to 8" << std::endl;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::TransferBits, 8);
            break;
        case '2':
            std::cout << "Setting bits to 16" << std::endl;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::TransferBits, 16);
            break;
        }

        double endFrameTime = getAbsoluteTime();
        totalTime += endFrameTime - startFrameTime;
        if (totalTime > 1.0)
        {
            lastProcessingFPS = numFrames / totalProcessedTime;
            cameraFPS = numFrames / cameraTime;
            std::cout << "Framerate: " << lastProcessingFPS << " fps" << std::endl;
            totalTime = 0.0;
            cameraTime = 0.0;
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

inline void drawBoxes(const cv::Mat &frame)
{
    if (frame.elemSize1() == 1)
    {
        cv::rectangle(frame, tempFrameBox, cv::Scalar(255, 0, 255), 2);
    }
    else
    {
        cv::rectangle(frame, tempFrameBox, cv::Scalar(65535, 0, 65535), 2);
    }
    if (isBoxSelected)
    {
        if (frame.elemSize1() == 1)
        {
            cv::rectangle(frame, fullFrameBox, cv::Scalar(0, 0, 255), 2);
        }
        else
        {
            cv::rectangle(frame, fullFrameBox, cv::Scalar(0, 0, 65535), 2);
        }
    }
}

void writeText(const cv::Mat _frame, std::string _text, int _line)
{
    const std::string fontFamily = "Arial";
    const cv::Scalar color{0, 200, 200, 0};
    const int fontSize = 40;
    const int fontSpacing = 15;
    const int height = _line * (fontSize + fontSpacing);

    cv::addText(_frame, _text, cv::Point(fontSpacing, height), fontFamily, fontSize, color);
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

    // qhyCamera.setStreamMode(sky360lib::camera::QHYCamera::SingleFrame);

    // check color camera
    if (qhyCamera.getCameraInfo()->isColor)
    {
        qhyCamera.setControl(sky360lib::camera::QHYCamera::RedWB, 70.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::GreenWB, 65.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::BlueWB, 78.0);
    }
    return true;
}

inline bool getQhyCameraImage(cv::Mat &cameraFrame)
{
    EASY_FUNCTION(profiler::colors::Purple);

    return qhyCamera.getFrame(cameraFrame, false);
}

inline void debayerImage(const cv::Mat &imageIn, cv::Mat &imageOut)
{
    EASY_FUNCTION(profiler::colors::Yellow);

    qhyCamera.debayerImage(imageIn, imageOut);
}

bool openVideo(const cv::Mat &frame, double meanFps)
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    auto name = "vo" + oss.str() + ".mkv";
    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    return videoWriter.open(name, codec, meanFps, frame.size(), true);
}