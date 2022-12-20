#include <iostream>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/tracking.hpp>

#include "bgs.hpp"
#include "tracker.hpp"
#include "profiling.hpp"

using namespace sky360lib::bgs;
using namespace sky360lib::tracking;

int main(int argc, const char **argv)
{
    cv::VideoCapture cap;
    //WeightedMovingVariance wmv;
    WeightedMovingVarianceHalide wmv;
    // cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();
    Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
    // Vibe vibeBGS;

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

    cv::Mat frame, greyFrame, bgImg;
    long numFrames = 0;
    double totalTime = 0;

    cap.read(frame);
    if (frame.type() != CV_8UC3)
    {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    cv::Mat bgsMask(frame.size(), CV_8UC1);

    // Applying first time for initialization of algo
    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
    // vibeBGS.apply(greyFrame, bgsMask);
    wmv.apply(greyFrame, bgsMask);

    cv::Rect boundingBox = cv::Rect(0, 0, greyFrame.size().width, greyFrame.size().height);
    tracker->init(bgsMask, boundingBox);

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
        cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);

        wmv.apply(greyFrame, bgsMask);
        double startTime = getAbsoluteTime();
        if (tracker->update(bgsMask, boundingBox))
        {
            cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
        }
        // vibeBGS.apply(greyFrame, bgsMask);
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

    return 0;
}
