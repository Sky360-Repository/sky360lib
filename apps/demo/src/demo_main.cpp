#include <iostream>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "bgs.hpp"
#include "profiling.hpp"

using namespace sky360lib::bgs;

int main(int argc, const char** argv) {
    cv::VideoCapture cap;
    //WeightedMovingVariance wmv;
    WeightedMovingVarianceHalide wmv;
    //Vibe vibeBGS;

    // if (argc < 2) {
    //     std::cout << "Need one parameter as camera number" << std::endl;
    //     return -1;
    // }

    double freq = initFrequency();

    // int camNum = std::stoi(argv[1]);
    //cap.open(0);//camNum);
    #ifdef _WIN32
    //cap.open("E:\\source\\sky360\\embedded-bgsub\\Dahua-20220901-184734.mp4");
    cap.open("E:\\source\\sky360\\dataset\\plane_flying_past2.mkv");
    #else
    cap.open("Dahua-20220901-184734.mp4");
    #endif
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
    if (frame.type() != CV_8UC3) {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    cv::Mat bgsMask(frame.size(), CV_8UC1);

    // Applying first time for initialization of algo
    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
    //vibeBGS.apply(greyFrame, bgsMask);
    wmv.apply(greyFrame, bgsMask);

    cv::imshow("BGS Demo", frame);

    std::cout << "Enter loop" << std::endl;
    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            std::cout << "No image" << std::endl;
            break;
        }
        cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);

        double startTime = getAbsoluteTime();
        wmv.apply(greyFrame, bgsMask);
        //vibeBGS.apply(greyFrame, bgsMask);
        double endTime = getAbsoluteTime();
        totalTime += endTime - startTime;
        ++numFrames;

        if (numFrames % 100 == 0) {
            std::cout << "Framerate: " << (numFrames / totalTime) << " fps" << std::endl;
            totalTime = 0;
            numFrames = 0;
        }
        cv::imshow("BGS Demo", bgsMask);
        cv::resizeWindow("BGS Demo", 1024, 1024);
        // cv::imshow("Live Video", frame);

        if ((char)cv::waitKey(1) == 27) {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
    }
    std::cout << "Exit loop\n" << std::endl;

    return 0;
}
