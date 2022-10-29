#include <iostream>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "bgs.hpp"
#include "profiling.hpp"

using namespace sky360lib::bgs;

int main(int argc, const char** argv) {
    cv::VideoCapture cap;
    WeightedMovingVariance wmv;
    Vibe vibeBGS;

    if (argc < 2) {
        std::cout << "Need one parameter as camera number" << std::endl;
        return -1;
    }

    double freq = initFrequency();

    int camNum = std::stoi(argv[1]);
    //cap.open(camNum);
    //cap.open("E:\\source\\sky360\\embedded-bgsub\\Dahua-20220901-184734.mp4");
    cap.open("E:\\source\\sky360\\dataset\\plane_flying_past2.mkv");
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
    //cv::namedWindow("BG Video", 0);

    cv::Mat frame, greyFrame, bgImg;
    long numFrames = 0;
    double totalTime = 0;

    cap >> frame;
    if (frame.type() != CV_8UC3) {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    //cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
    //vibeBGS.initialize(greyFrame, 12);
    std::cout << "initializeParallel" << std::endl;

    cv::Mat bgsMask(frame.size(), CV_8UC1);

    cv::imshow("BGS Demo", frame);

    std::cout << "Enter loop" << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "No image" << std::endl;
            break;
        }
        cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);

        double startTime = getAbsoluteTime();
        wmv.process(greyFrame, bgsMask);
        //vibeBGS.apply(greyFrame, bgsMask);
        double endTime = getAbsoluteTime();
        totalTime += endTime - startTime;
        ++numFrames;
        //std::cout << "Frame: " << numFrames << std::endl;

        if (numFrames % 100 == 0) {
            std::cout << "Framerate: " << (numFrames / totalTime) << " fps" << std::endl;
        }
        cv::imshow("BGS Demo", bgsMask);
        cv::imshow("Live Video", frame);
        //vibeBGS.getBackgroundImage(bgImg);
        //cv::imshow("BG Video", bgImg);

        char c = (char)cv::waitKey(10);
        if (c == 27) {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
    }
    std::cout << "Exit loop\n" << std::endl;

    return 0;
}
