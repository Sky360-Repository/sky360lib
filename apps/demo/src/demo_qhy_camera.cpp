#include <iostream>
#include <string>
#include <algorithm>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "../../../api/camera/qhy_camera.hpp"
#include "../../../api/utils/profiler.hpp"
#include "../../../api/bgs/bgs.hpp"
#include "../../../api/blobs/connectedBlobDetection.hpp"

/////////////////////////////////////////////////////////////
// Default parameters
const int DEFAULT_WINDOW_WIDTH{1024};
int blur_radius{3};
bool applyGreyscale{false};
bool applyNoiseReduction{true};
bool isVideoOpen = false;
cv::VideoWriter videoWriter;
sky360lib::utils::Profiler profiler;

/////////////////////////////////////////////////////////////
// Background subtractor to use
enum BGSType
{
    NoBGS
    ,Vibe
    ,WMV
};
std::unique_ptr<sky360lib::bgs::CoreBgs> bgsPtr{nullptr};

/////////////////////////////////////////////////////////////
// Blob Detector
sky360lib::blobs::ConnectedBlobDetection blobDetector;

/////////////////////////////////////////////////////////////
// Camera
sky360lib::camera::QhyCamera qhyCamera;

/////////////////////////////////////////////////////////////
// Function Definitions
void writeText(const cv::Mat _frame, std::string _text, int _line);
std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type);
std::string getBGSName(BGSType _type);
inline void appyPreProcess(const cv::Mat &input, cv::Mat &output);
inline void appyBGS(const cv::Mat &input, cv::Mat &output);
inline void drawBboxes(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &frame);
inline std::vector<cv::Rect> findBlobs(const cv::Mat &image);
inline void drawBboxes(std::vector<cv::Rect> &keypoints, const cv::Mat &frame);
inline void outputBoundingBoxes(std::vector<cv::Rect> &bboxes);
bool openQQYCamera();
inline bool getQhyCameraImage(cv::Mat &cameraFrame);
bool openVideo(const cv::Mat &frame, double meanFps);
inline void debayerImage(const cv::Mat &imageIn, cv::Mat &imageOut);

static void changeParam(int value, void* paramP)
{
    long param = (long)paramP;
    switch (param)
    {
        case sky360lib::camera::QhyCamera::ControlParam::Gain:
            qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Gain, (double)value);
            break;
    }
}

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    BGSType bgsType{WMV};

    blobDetector.set_min_distance(40);

    if (!openQQYCamera())
    {
        return -1;
    }
    std::cout << qhyCamera.get_camera_info()->to_string() << std::endl;

    double aspectRatio{(double)qhyCamera.get_camera_info()->chip.max_image_width / (double)qhyCamera.get_camera_info()->chip.max_image_height};

    double exposure = (argc > 1 ? atoi(argv[1]) : 20000);
    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure);

    const auto concurrentThreads = std::thread::hardware_concurrency();
    std::cout << "Available number of concurrent threads = " << concurrentThreads << std::endl;

    bgsPtr = createBGS(bgsType);

    if (cv::ocl::haveOpenCL())
    {
        std::cout << "Has OpenCL support, using it on OpenCV" << std::endl;
    }

    cv::namedWindow("BGS", cv::WINDOW_NORMAL);
    cv::namedWindow("Live Video", cv::WINDOW_NORMAL);

    cv::resizeWindow("BGS", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_WIDTH / aspectRatio);
    cv::resizeWindow("Live Video", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_WIDTH / aspectRatio);

    int gain = 30;
    cv::createTrackbar("Gain:", "Live Video", nullptr, 50, changeParam, (void*)(long)sky360lib::camera::QhyCamera::ControlParam::Gain);
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

    cv::Mat bgsMask{frame.size(), CV_8UC1};
    cv::Mat videoFrame{frame.size(), CV_8UC3};

    std::vector<cv::Rect> bboxes;
    bool run = true;
    bool pause = false;
    bool doBlobDetection = false;
    std::cout << "Enter loop" << std::endl;
    while (run)
    {
        auto startFrameTime = std::chrono::high_resolution_clock::now();

        if (!pause)
        {
            auto startProcessedTime = std::chrono::high_resolution_clock::now();
            profiler.start("GetImage");
            getQhyCameraImage(frame);
            profiler.stop("GetImage");
            cameraTime += qhyCamera.get_last_frame_capture_time();
            appyPreProcess(frame, processedFrame);
            appyBGS(processedFrame, bgsMask);
            if (doBlobDetection)
            {
                bboxes = findBlobs(bgsMask);
            }
            auto endProcessedTime = std::chrono::high_resolution_clock::now();
            debayerImage(frame, frameDebayered);
            if (doBlobDetection)
            {
                // drawBboxes(bboxes, bgsMask);
                drawBboxes(bboxes, frameDebayered);
            }

            if (frameDebayered.elemSize1() > 1)
            {
                frameDebayered.convertTo(videoFrame, CV_8U, 1 / 256.0f);
            }
            else
            {
                videoFrame = frameDebayered;
            }
            writeText(videoFrame, "Exposure: " + std::to_string(exposure / 1000.0) + " ms ('+' to +10%, '-' to -10%)", 1);
            writeText(videoFrame, "Capture: " + std::to_string(cameraFPS) + " fps", 2);
            writeText(videoFrame, "Total Processing: " + std::to_string(lastProcessingFPS) + " fps", 3);
            writeText(videoFrame, "Blob Detection: " + std::string(doBlobDetection ? "On" : "Off") + " ('b' to toggle)", 4);
            writeText(videoFrame, "Video Recording: " + std::string(isVideoOpen ? "Yes" : "No") + " ('v' to toggle)", 5);
            writeText(videoFrame, "BGS: " + getBGSName(bgsType) + " ('s' to toggle)", 6);
            writeText(videoFrame, "Bits: " + std::to_string(qhyCamera.get_camera_params().bpp) + " ('1' to 8 bits, '2' to 16 bits)", 7);

            ++numFrames;
            totalProcessedTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endProcessedTime - startProcessedTime).count() * 1e-9;
            totalMeanProcessedTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endProcessedTime - startProcessedTime).count() * 1e-9;
            ++totalNumFrames;
            cv::imshow("BGS", bgsMask);
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
                outputBoundingBoxes(bboxes);
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
            case 'b':
                doBlobDetection = !doBlobDetection;
                std::cout << "Blob Detection: " << doBlobDetection << std::endl;
                break;
            case 's':
                bgsType = bgsType == BGSType::WMV ? BGSType::Vibe : (bgsType == BGSType::Vibe ? BGSType::NoBGS : BGSType::WMV);
                bgsPtr = createBGS(bgsType);
                std::cout << "Setting BGS to: " << std::to_string(bgsType) << std::endl;
                break;
            case '+':
                exposure *= 1.1;
                std::cout << "Setting exposure to: " << exposure << std::endl;
                qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure);
                break;
            case '-':
                exposure *= 0.9;
                std::cout << "Setting exposure to: " << exposure << std::endl;
                qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure);
                break;
            case '1':
                std::cout << "Setting bits to 8" << std::endl;
                qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::TransferBits, 8);
                bgsPtr = createBGS(bgsType);
                break;
            case '2':
                std::cout << "Setting bits to 16" << std::endl;
                qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::TransferBits, 16);
                bgsPtr->restart();
                break;
        }

        auto endFrameTime = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endFrameTime - startFrameTime).count() * 1e-9;
        if (totalTime > 1.0)
        {
            lastProcessingFPS = numFrames / totalProcessedTime;
            cameraFPS = numFrames / cameraTime;
            std::cout << "Framerate: " << lastProcessingFPS << " fps" << std::endl;
            totalTime = 0.0;
            cameraTime = 0.0;
            totalProcessedTime = 0.0;
            numFrames = 0;
            profiler.report();
        }
    }
    std::cout << "Exit loop\n"
              << std::endl;
    std::cout << std::endl
              << "Average Framerate: " << (totalNumFrames / totalMeanProcessedTime) << " fps" << std::endl;

    cv::destroyAllWindows();

    qhyCamera.close();

    return 0;
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

std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type)
{
    switch (_type)
    {
    case BGSType::Vibe:
        return std::make_unique<sky360lib::bgs::Vibe>(sky360lib::bgs::VibeParams(50, 24, 1, 2));
    case BGSType::WMV:
        return std::make_unique<sky360lib::bgs::WeightedMovingVariance>();
    default:
        return nullptr;
    }
}

std::string getBGSName(BGSType _type)
{
    switch (_type)
    {
        case NoBGS: return "No BGS";
        case Vibe: return "Vibe";
        case WMV: return "Weighted Moving Variance";
    }
    return "ERROR!";
}

inline void appyPreProcess(const cv::Mat &input, cv::Mat &output)
{
    cv::Mat tmpFrame;

    if (applyGreyscale && input.channels() > 1)
    {
        cv::cvtColor(input, tmpFrame, cv::COLOR_RGB2GRAY);
    }
    else
    {
        tmpFrame = input;
    }
    if (applyNoiseReduction)
    {
        cv::GaussianBlur(tmpFrame, output, cv::Size(blur_radius, blur_radius), 0);
    }
    else
    {
        output = tmpFrame;
    }
}

inline void appyBGS(const cv::Mat &input, cv::Mat &output)
{
    if (bgsPtr != nullptr)
    {
        bgsPtr->apply(input, output);
    }
    else
    {
        output = input;
    }
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
        if (frame.elemSize1() == 1)
        {
            cv::rectangle(frame, bb, cv::Scalar(255, 0, 255), 2);
        }
        else
        {
            cv::rectangle(frame, bb, cv::Scalar(65535, 0, 65535), 2);
        }
    }
}

// Finds the connected components in the image and returns a list of bounding boxes
inline std::vector<cv::Rect> findBlobs(const cv::Mat &image)
{
    std::vector<cv::Rect> blobs;
    blobDetector.detect(image, blobs);

    return blobs;
}

bool openQQYCamera()
{
    auto cameras = qhyCamera.get_cameras();
    if (cameras.size() == 0)
    {
        return false;
    }
    if (!qhyCamera.open(cameras.begin()->first))
    {
        std::cout << "Error opening camera" << std::endl;
        return false;
    }

    return true;
}

inline bool getQhyCameraImage(cv::Mat &cameraFrame)
{
    return qhyCamera.get_frame(cameraFrame, false);
}

inline void debayerImage(const cv::Mat &imageIn, cv::Mat &imageOut)
{
    qhyCamera.debayer_image(imageIn, imageOut);
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