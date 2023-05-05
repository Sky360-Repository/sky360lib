#include <iostream>
#include <string>

#include "qhyCamera.hpp"
#include "utils.hpp"
#include "autoExposureControl.hpp"
#include "profiler.hpp"
#include "textWriter.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

/////////////////////////////////////////////////////////////
// Variables
const int DEFAULT_WINDOW_WIDTH{1024};
const int DEFAULT_BOX_SIZE{500};

bool isVideoOpen = false;
bool isBoxSelected = false;
cv::Size frameSize;
double clipLimit = 2.0;
bool doEqualization = false;
bool doAutoExposure = false;
bool squareResolution = false;
bool run = true;
bool pauseCapture = false;
bool showHistogram = false;
bool settingCircle = false;
bool circleSet = false;

cv::Rect fullFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Rect tempFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Point2d circleInit;
cv::Point2d circleEnd;
cv::Point2d circleCenter;
double circleRadius{0.0f};
double cameraCircleMaxFov{0.0};

cv::VideoWriter videoWriter;
sky360lib::utils::DataMap profileData;
sky360lib::utils::Profiler profiler;
sky360lib::camera::QHYCamera qhyCamera;
sky360lib::utils::TextWriter textWriter;
sky360lib::utils::AutoExposureControl autoExposureControl;

/////////////////////////////////////////////////////////////
// Function Definitions
inline void drawBoxes(const cv::Mat &frame);
bool openQQYCamera();
bool openVideo(const cv::Size &size, double meanFps);
void createControlPanel();
void treatKeyboardpress(char key);
void changeTrackbars(int value, void *paramP);
void mouseCallBackFunc(int event, int x, int y, int, void *);
void exposureCallback(int, void*userData);
void TransferbitsCallback(int, void*userData);
void generalCallback(int, void*userData);
void drawFOV(cv::Mat& frame, double max_fov, cv::Point2d center, double radius);

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    if (!openQQYCamera())
    {
        return -1;
    }
    std::cout << qhyCamera.getCameraInfo()->toString() << std::endl;

    double exposure = (argc > 1 ? atoi(argv[1]) : 20000);
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);

    double gain = 5.0;
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Gain, gain);

    int frame_counter = 0;
    int auto_exposure_frame_interval = 3; 

    createControlPanel();

    cv::Mat frame, processedFrame, saveFrame, frameDebayered;

    qhyCamera.getFrame(frame, false);
    frameSize = frame.size();

    cv::Mat videoFrame{frame.size(), CV_8UC3};

    std::vector<cv::Rect> bboxes;
    std::cout << "Enter loop" << std::endl;
    while (run)
    {
        profiler.start("Frame");
        if (!pauseCapture)
        {
            profiler.start("GetImage");
            qhyCamera.getFrame(frame, false);
            profiler.stop("GetImage");
            frameSize = frame.size();
            profiler.start("Debayer");
            qhyCamera.debayerImage(frame, frameDebayered);
            profiler.stop("Debayer");
            if (doEqualization)
            {
                profiler.start("Equalization");
                sky360lib::utils::Utils::equalizeImage(frameDebayered, frameDebayered, clipLimit);
                profiler.stop("Equalization");
            }

            if (doAutoExposure)
            {
                frame_counter++;

                if (frame_counter % auto_exposure_frame_interval == 0) { // to improve fps
                    std::pair<double, double> exposure_gain = autoExposureControl.calculate_exposure_gain(frameDebayered, exposure, gain);
                    exposure = exposure_gain.first;
                    gain = exposure_gain.second;
                    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
                    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Gain, gain);
                }
            }

            if (isBoxSelected)
            {
                cv::Mat cropFrame = frameDebayered(fullFrameBox);
                cv::imshow("Window Cut", cropFrame);
            }

            drawBoxes(frameDebayered);
            if (!squareResolution)
            {
                drawFOV(frameDebayered, 220.0, circleCenter, circleRadius);
            }
            else
            {
                drawFOV(frameDebayered, cameraCircleMaxFov, cv::Point(frameDebayered.size().width / 2, frameDebayered.size().height / 2), frameDebayered.size().width / 2);
            }
            exposure = (double)qhyCamera.getCameraParams().exposureTime;
            textWriter.writeText(frameDebayered, "Exposure: " + sky360lib::utils::Utils::formatDouble(exposure / 1000.0, 2) + " ms ('+' to +10%, '-' to -10%)", 1);
            textWriter.writeText(frameDebayered, "Gain: " + std::to_string(qhyCamera.getCameraParams().gain), 2);
            textWriter.writeText(frameDebayered, "Resolution: " + std::to_string(qhyCamera.getCameraParams().roiWidth) + " x " + std::to_string(qhyCamera.getCameraParams().roiHeight), 3);
            textWriter.writeText(frameDebayered, "Bits: " + std::to_string(qhyCamera.getCameraParams().transferBits) + " ('1' to 8 bits, '2' to 16 bits)", 4);
            textWriter.writeText(frameDebayered, "Image Equalization: " + std::string(doEqualization ? "On" : "Off") + " ('e' to toggle)", 5);
            textWriter.writeText(frameDebayered, "Auto Exposure: " + std::string(doAutoExposure ? "On" : "Off") + " ('a' to toggle)", 6);
            textWriter.writeText(frameDebayered, "TargetMSV: " + sky360lib::utils::Utils::formatDouble(autoExposureControl.get_targetMSV()) + " '+' to +10%, '-' to -10%", 7);
            textWriter.writeText(frameDebayered, "Video Recording: " + std::string(isVideoOpen ? "Yes" : "No") + " ('v' to toggle)", 8);
            textWriter.writeText(frameDebayered, "Max Capture FPS: " + sky360lib::utils::Utils::formatDouble(profileData["GetImage"].fps(), 2), 1, true);
            textWriter.writeText(frameDebayered, "Frame FPS: " + sky360lib::utils::Utils::formatDouble(profileData["Frame"].fps(), 2), 2, true);

            cv::imshow("Live Video", frameDebayered);
            if (showHistogram){
                cv::Mat hist = sky360lib::utils::Utils::createHistogram(frameDebayered);
                cv::imshow("Histogram", hist);
            }
            if (isVideoOpen)
            {
                if (frameDebayered.elemSize1() > 1)
                {
                    frameDebayered.convertTo(videoFrame, CV_8U, 1 / 256.0f);
                }
                else
                {
                    videoFrame = frameDebayered;
                }
                videoWriter.write(videoFrame);
            }
        }

        treatKeyboardpress((char)cv::waitKey(1)); 

        profiler.stop("Frame");
        if (profiler.getData("Frame").durationInSeconds() > 1.0)
        {
            profileData = profiler.getData();
            profiler.reset();
        }
    }
    std::cout << "Exit loop\n"
              << std::endl;

    //cv::destroyAllWindows();

    qhyCamera.close();

    return 0;
}

void createControlPanel()
{
    double aspectRatio = (double)qhyCamera.getCameraInfo()->maxImageWidth / (double)qhyCamera.getCameraInfo()->maxImageHeight;
    cv::namedWindow("Live Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live Video", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_WIDTH / aspectRatio);
    cv::setMouseCallback("Live Video", mouseCallBackFunc, NULL);

    int maxUsbTraffic = (int)qhyCamera.getCameraInfo()->usbTrafficLimits.max;
    cv::createTrackbar("USB Traffic:", "", nullptr, maxUsbTraffic, changeTrackbars, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::UsbTraffic);
    cv::setTrackbarPos("USB Traffic:", "", (int)qhyCamera.getCameraParams().usbTraffic);
    cv::createButton("0.1 ms", exposureCallback, (void *)(long)100, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("1 ms", exposureCallback, (void *)(long)1000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("10 ms", exposureCallback, (void *)(long)10000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("100 ms", exposureCallback, (void *)(long)100000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("1 s", exposureCallback, (void *)(long)1000000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("- 10%", exposureCallback, (void *)(long)-2, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("+ 10%", exposureCallback, (void *)(long)-1, cv::QT_PUSH_BUTTON, 1);
    int maxGain = (int)qhyCamera.getCameraInfo()->gainLimits.max;
    cv::createTrackbar("Gain:", "", nullptr, maxGain, changeTrackbars, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::Gain);
    cv::setTrackbarPos("Gain:", "", (int)qhyCamera.getCameraParams().gain);
    int maxOffset = (int)qhyCamera.getCameraInfo()->offsetLimits.max;
    cv::createTrackbar("Offset:", "", nullptr, maxOffset, changeTrackbars, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::Offset);
    cv::setTrackbarPos("Offset:", "", (int)qhyCamera.getCameraParams().offset);
    cv::createButton("8 bits", TransferbitsCallback, (void *)(long)8, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("16 bits", TransferbitsCallback, (void *)(long)16, cv::QT_PUSH_BUTTON, 1);

    int maxRedWB = (int)qhyCamera.getCameraInfo()->redWBLimits.max;
    cv::createTrackbar("Red WB:", "", nullptr, maxRedWB, changeTrackbars, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::RedWB);
    cv::setTrackbarPos("Red WB:", "", (int)qhyCamera.getCameraParams().redWB);
    int maxGreenWB = (int)qhyCamera.getCameraInfo()->greenWBLimits.max;
    cv::createTrackbar("Green WB:", "", nullptr, maxGreenWB, changeTrackbars, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::GreenWB);
    cv::setTrackbarPos("Green WB:", "", (int)qhyCamera.getCameraParams().greenWB);
    int maxBlueWB = (int)qhyCamera.getCameraInfo()->blueWBLimits.max;
    cv::createTrackbar("Blue WB:", "", nullptr, maxBlueWB, changeTrackbars, (void *)(long)sky360lib::camera::QHYCamera::ControlParam::BlueWB);
    cv::setTrackbarPos("Blue WB:", "", (int)qhyCamera.getCameraParams().blueWB);

    cv::createButton("AE on/off", generalCallback, (void *)(long)'a', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Square Res. on/off", generalCallback, (void *)(long)'s', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Image Equalization", generalCallback, (void *)(long)'e', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Video Recording", generalCallback, (void *)(long)'v', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Histogram on/off", generalCallback, (void *)(long)'h', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Exit Program", generalCallback, (void *)(long)27, cv::QT_PUSH_BUTTON, 1);
}

void treatKeyboardpress(char key)
{
    switch (key)
    {
    case 27:
        run = false;
        break;
    case ' ':
        std::cout << "Pausing" << std::endl;
        pauseCapture = !pauseCapture;
        break;
    case 'e':
        doEqualization = !doEqualization;
        break;
    case 'v':
        if (!isVideoOpen)
        {
            std::cout << "Start recording" << std::endl;
            isVideoOpen = openVideo(frameSize, profileData["Frame"].fps());
        }
        else
        {
            std::cout << "End recording" << std::endl;
            isVideoOpen = false;
            videoWriter.release();
        }
        break;
    case '+':
        {
            double exposure = (double)qhyCamera.getCameraParams().exposureTime * 1.1;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
            if(doAutoExposure)
            {
                double targetMSV = autoExposureControl.get_targetMSV();
                autoExposureControl.set_targetMSV(targetMSV * 1.1);
            }
        }
        break;
    case '-':
        {
            double exposure = (double)qhyCamera.getCameraParams().exposureTime * 0.9;
            qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
            if(doAutoExposure)
            {
                double targetMSV = autoExposureControl.get_targetMSV();
                autoExposureControl.set_targetMSV(targetMSV * 0.9);
            }
        }
        break;
    case '1':
        std::cout << "Setting bits to 8" << std::endl;
        qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::TransferBits, 8);
        break;
    case '2':
        std::cout << "Setting bits to 16" << std::endl;
        qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::TransferBits, 16);
        break;
    case 's':
        squareResolution = !squareResolution;
        isBoxSelected = false;
        if (squareResolution)
        {
            if (circleSet)
            {
                double max_radius = std::min(std::min(circleCenter.y, qhyCamera.getCameraInfo()->maxImageHeight - circleCenter.y), circleRadius);
                cameraCircleMaxFov = (max_radius / circleRadius) * 220.0;

                uint32_t width = ((uint32_t)max_radius * 2);
                uint32_t height = width;
                uint32_t x = (uint32_t)(circleCenter.x - (width / 2)) & ~0x1;
                uint32_t y = (uint32_t)(circleCenter.y - (height / 2)) & ~0x1;

                qhyCamera.setResolution(x, y, width, height);
            }
            else
            {
                uint32_t x = ((uint32_t)qhyCamera.getCameraInfo()->maxImageWidth - (uint32_t)qhyCamera.getCameraInfo()->maxImageHeight) / 2;
                uint32_t y = 0;
                uint32_t width = qhyCamera.getCameraInfo()->maxImageHeight;
                uint32_t height = qhyCamera.getCameraInfo()->maxImageHeight;

                qhyCamera.setResolution(x, y, width, height);
            }
        }
        else
        {
            uint32_t x = 0;
            uint32_t y = 0;
            uint32_t width = qhyCamera.getCameraInfo()->maxImageWidth;
            uint32_t height = qhyCamera.getCameraInfo()->maxImageHeight;
            qhyCamera.setResolution(x, y, width, height);
        }
        break;
    case 'h':
        showHistogram = !showHistogram;
        if (!showHistogram)
        {
            cv::destroyWindow("Histogram");
        }
        break;
    case 'a':
        doAutoExposure = !doAutoExposure;
        break;
    }

}

void changeTrackbars(int value, void *paramP)
{
    long param = (long)paramP;
    qhyCamera.setControl((sky360lib::camera::QHYCamera::ControlParam)param, (double)value);
}

void exposureCallback(int, void*userData)
{
    double exposure = (double)(long)userData;

    if(doAutoExposure)
    {
        if ((long)userData == -1)
        {
            double targetMSV = autoExposureControl.get_targetMSV();
            autoExposureControl.set_targetMSV(targetMSV * 1.1);
        }
        else if ((long)userData == -2)
        {
            double targetMSV = autoExposureControl.get_targetMSV();
            autoExposureControl.set_targetMSV(targetMSV * 0.9);
        }
        return;
    }

    if ((long)userData == -1)
    {
        exposure = (double)qhyCamera.getCameraParams().exposureTime * 1.1;
    }
    else if ((long)userData == -2)
    {
        exposure = (double)qhyCamera.getCameraParams().exposureTime * 0.9;
    }
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::Exposure, exposure);
}

void TransferbitsCallback(int, void*userData)
{
    long transferBits = (long)userData;
    qhyCamera.setControl(sky360lib::camera::QHYCamera::ControlParam::TransferBits, transferBits);
}

void generalCallback(int, void*userData)
{
    long param = (long)userData;
    treatKeyboardpress((char)param);
}

void drawOneFov(cv::Mat& frame, cv::Point2d center, double fov, double max_fov, const cv::Scalar& color)
{
    if (max_fov >= fov)
    {
        cv::Point2f text_offset(25, 60);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 2.6;
        int font_thickness = 7;

        double radius = circleRadius * (fov / 220.0);
        cv::circle(frame, center, radius, color, 8);

        std::string fov_text = sky360lib::utils::Utils::formatDouble(fov, 2);
        cv::Size text_size = cv::getTextSize(fov_text, font_face, font_scale, font_thickness, nullptr);
        double textX = std::max(center.x - radius + text_offset.x, 0.0);
        cv::Point2f text_position(textX, center.y + text_size.height / 2 + text_offset.y);
        cv::putText(frame, fov_text, text_position, font_face, font_scale, color, font_thickness, cv::LINE_AA);
    }
}

void drawFOV(cv::Mat& frame, double max_fov, cv::Point2d center, double radius)
{
    if (settingCircle || circleSet)
    {
        cv::Scalar color;
        if (frame.elemSize1() == 1)
        {
            color = cv::Scalar(128, 128, 255);
        }
        else
        {
            color = cv::Scalar(32767, 32767, 65535);
        }
        cv::line(frame, cv::Point2d(center.x, center.y - radius), cv::Point2d(center.x, center.y + radius), color, 8);
        cv::line(frame, cv::Point2d(center.x - radius, center.y), cv::Point2d(center.x + radius, center.y), color, 8);

        drawOneFov(frame, center, 220.0, max_fov, color);
        drawOneFov(frame, center, 180.0, max_fov, color);
        drawOneFov(frame, center, 90.0, max_fov, color);
        drawOneFov(frame, center, 30.0, max_fov, color);
        drawOneFov(frame, center, 0.0f, max_fov, color);

        double max_radius = std::min(std::min(center.y, frame.size().height - center.y), radius);
        double circleMaxFov = (max_radius / radius) * max_fov;
        drawOneFov(frame, center, circleMaxFov, max_fov, color);
        //std::cout << "max_radius: " << max_radius << ", radius: " << radius << ", circleMaxFov: " << circleMaxFov << ", max_fov: " << max_fov << std::endl;
    }
}

void mouseCallBackFunc(int event, int x, int y, int flags, void *)
{
    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN:
            if (flags & cv::EVENT_FLAG_SHIFTKEY)
            {
                settingCircle = true;
                circleInit.x = x;
                circleInit.y = y;
            }
            break;
        case cv::EVENT_LBUTTONUP:
            if (flags & cv::EVENT_FLAG_SHIFTKEY)
            {
                settingCircle = false;
                circleSet = true;
            }
            else
            {
                fullFrameBox = tempFrameBox;
                isBoxSelected = true;
            }
            break;
        case cv::EVENT_MOUSEMOVE:
            if (!settingCircle)
            {
                //tempFrameBox
                if (x > (frameSize.width - (tempFrameBox.width / 2.0)))
                {
                    x = frameSize.width - (tempFrameBox.width / 2.0);
                }
                else if (x < (tempFrameBox.width / 2.0))
                {
                    x = (tempFrameBox.width / 2.0);
                }
                if (y > (frameSize.height - (tempFrameBox.height / 2.0)))
                {
                    y = frameSize.height - (tempFrameBox.height / 2.0);
                }
                else if (y < (tempFrameBox.height / 2.0))
                {
                    y = (tempFrameBox.height / 2.0);
                }
                tempFrameBox.x = x - (tempFrameBox.width / 2.0);
                tempFrameBox.y = y - (tempFrameBox.height / 2.0);
            }
            else
            {
                circleEnd.x = x;
                circleEnd.y = y;

                circleCenter.x = std::abs((circleInit.x + circleEnd.x) / 2);
                circleCenter.y = std::abs((circleInit.y + circleEnd.y) / 2);
                circleRadius = std::sqrt((circleCenter.x - circleInit.x) * (circleCenter.x - circleInit.x) + (circleCenter.y - circleInit.y) * (circleCenter.y - circleInit.y));
            }
            break;
    }
}

inline void drawBoxes(const cv::Mat &frame)
{
    if (frame.elemSize1() == 1)
    {
        cv::rectangle(frame, tempFrameBox, cv::Scalar(255, 0, 255), 5);
    }
    else
    {
        cv::rectangle(frame, tempFrameBox, cv::Scalar(65535, 0, 65535), 5);
    }
    if (isBoxSelected)
    {
        if (frame.elemSize1() == 1)
        {
            cv::rectangle(frame, fullFrameBox, cv::Scalar(0, 0, 255), 5);
        }
        else
        {
            cv::rectangle(frame, fullFrameBox, cv::Scalar(0, 0, 65535), 5);
        }
    }
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

    if (qhyCamera.getCameraInfo()->isColor)
    {
        qhyCamera.setControl(sky360lib::camera::QHYCamera::RedWB, 180.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::GreenWB, 128.0);
        qhyCamera.setControl(sky360lib::camera::QHYCamera::BlueWB, 190.0);
    }
    return true;
}

bool openVideo(const cv::Size &size, double meanFps)
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    auto name = "vo" + oss.str() + ".mkv";
    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    return videoWriter.open(name, codec, meanFps, size, true);
}
