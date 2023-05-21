#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "qhy_camera.hpp"
#include "utils.hpp"
#include "autoExposureControl.hpp"
#include "autoWhiteBalance.hpp"
#include "profiler.hpp"
#include "textWriter.hpp"
#include "bgs.hpp"
#include "ringbuf.h"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

enum BGSType
{
    NoBGS
    ,Vibe
    ,WMV
};

/////////////////////////////////////////////////////////////
// Variables
const int DEFAULT_WINDOW_WIDTH{1024};
const int DEFAULT_BOX_SIZE{500};

bool isVideoOpen = false;
bool isBoxSelected = false;
cv::Size frameSize;
double clipLimit = 4.0;
bool doEqualization = false;
bool doAutoExposure = false;
bool doAutoWhiteBalance = false;
bool squareResolution = false;
bool run = true;
bool pauseCapture = false;
bool showHistogram = false;
bool settingCircle = false;
bool circleSet = false;
BGSType bgsType{NoBGS};

cv::Rect fullFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Rect tempFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Point2d circleInit;
cv::Point2d circleEnd;
cv::Point2d circleCenter;
double circleRadius{0.0f};
double cameraCircleMaxFov{0.0};
baudvine::RingBuf<double, 200> noise_buffer;
baudvine::RingBuf<double, 200> sharpness_buffer;

cv::VideoWriter videoWriter;
sky360lib::utils::DataMap profileData;
sky360lib::utils::Profiler profiler;
sky360lib::camera::QhyCamera qhyCamera;
sky360lib::utils::TextWriter textWriter;
sky360lib::utils::TextWriter textWriterCrop(cv::Scalar{80, 140, 190, 0}, 24, 2.5);
sky360lib::utils::AutoExposureControl autoExposureControl;
sky360lib::utils::AutoWhiteBalance autoWhiteBalance;
std::unique_ptr<sky360lib::bgs::CoreBgs> bgsPtr{nullptr};


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
void log_changes(const std::string& log_file_name, const std::string& action, double msv, double targetMSV, double exposure, double gain, double noise_level, double entropy, double sharpness, double redWB, double blueWB, double greenWB);
std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type);
std::string getBGSName(BGSType _type);

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    if (!openQQYCamera())
    {
        return -1;
    }
    std::cout << qhyCamera.get_camera_info()->to_string() << std::endl;
    qhyCamera.set_debug_info(false);

    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, (argc > 1 ? atoi(argv[1]) : 20000));
    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Gain, 5.0);

    int frame_counter = 0;
    int auto_exposure_frame_interval = 3; 

    createControlPanel();

    cv::Mat frame, processedFrame, saveFrame, frameDebayered;

    qhyCamera.get_frame(frame, false);
    frameSize = frame.size();

    cv::Mat videoFrame{frame.size(), CV_8UC3};

    sky360lib::utils::WhiteBalanceValues wbValues;
    wbValues.blue = qhyCamera.get_camera_params().blue_white_balance;
    wbValues.green = qhyCamera.get_camera_params().green_white_balance;
    wbValues.red = qhyCamera.get_camera_params().red_white_balance;

    sky360lib::utils::WhiteBalanceValues defaultWbValues = wbValues;

    std::cout << "RedWB: " << wbValues.red << ", GreenWB: " << wbValues.green << ", BlueWB: " << wbValues.blue << std::endl;

    // double aeFPS = 0.0;
    double noise_level = 0.0;
    double entropy = 0.0;
    double sharpness = 0.0;

    double adjustmentFactor = 0.9;
    double previousExposure = qhyCamera.get_camera_params().exposure;
    const double exposureChangeThreshold = 0.15;

    std::vector<cv::Rect> bboxes;
    std::cout << "Enter loop" << std::endl;
    while (run)
    {
        profiler.start("Frame");
        if (!pauseCapture)
        {
            profiler.start("GetImage");
            qhyCamera.get_frame(frame, false);
            profiler.stop("GetImage");
            frameSize = frame.size();
            profiler.start("Debayer");
            qhyCamera.debayer_image(frame, frameDebayered);
            profiler.stop("Debayer");
            if (doEqualization)
            {
                profiler.start("Equalization");
                sky360lib::utils::Utils::equalize_image(frameDebayered, frameDebayered, clipLimit);
                profiler.stop("Equalization");
            }
            if (doAutoWhiteBalance)
            {
                double currentExposure = qhyCamera.get_camera_params().exposure;
                double exposureChange = std::abs(currentExposure - previousExposure) / previousExposure;

                if (exposureChange >= exposureChangeThreshold && currentExposure < autoExposureControl.get_max_exposure())
                {
                    wbValues = autoWhiteBalance.grayWorld(frameDebayered, wbValues, adjustmentFactor, 3);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::RedWB, wbValues.red);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::GreenWB, wbValues.green);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::BlueWB, wbValues.blue);

                    cv::setTrackbarPos("Red WB:", "", (int)wbValues.red);
                    cv::setTrackbarPos("Green WB:", "", (int)wbValues.green);
                    cv::setTrackbarPos("Blue WB:", "", (int)wbValues.blue);

                    previousExposure = currentExposure;
                    log_changes("auto_exposure_log.txt", "Update WB", autoExposureControl.get_current_msv(), autoExposureControl.get_target_msv(), currentExposure, qhyCamera.get_camera_params().gain, noise_level, entropy, sharpness, wbValues.red, wbValues.blue, wbValues.green);
                }
                
                if (currentExposure >= autoExposureControl.get_max_exposure())
                {
                    std::cout << "Switch to the generic white balance values" << std::endl;

                    // Switch to the generic white balance values
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::RedWB, defaultWbValues.red);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::GreenWB, defaultWbValues.green);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::BlueWB, defaultWbValues.blue);
        
                    // Log the switch to generic white balance
                    log_changes("auto_exposure_log.txt", "Switch to Generic WB", autoExposureControl.get_current_msv(), autoExposureControl.get_target_msv(), currentExposure, qhyCamera.get_camera_params().gain, noise_level, entropy, sharpness, defaultWbValues.red, defaultWbValues.green, defaultWbValues.blue);
                }
            }

            if (doAutoExposure)
            {
                frame_counter++;

                // to improve fps
                if (frame_counter % auto_exposure_frame_interval == 0) 
                { 
                    profiler.start("AutoExposure");
                    const double exposure = (double)qhyCamera.get_camera_params().exposure;
                    const double gain = (double)qhyCamera.get_camera_params().gain;
                    auto exposure_gain = autoExposureControl.calculate_exposure_gain(frame, exposure, gain);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure_gain.exposure);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Gain, exposure_gain.gain);                    
                   
                    // Log exposure update
                    if (exposure_gain.exposure != exposure) 
                    {
                        std::string action = "Update Exposure";
                        std::string log_file_name = "auto_exposure_log.txt"; // Replace with the desired log file name
                        log_changes(log_file_name, action, autoExposureControl.get_current_msv(), autoExposureControl.get_target_msv(), exposure_gain.exposure, gain, noise_level, entropy, sharpness, wbValues.red, wbValues.green, wbValues.blue);
                    }

                    // Log gain update
                    if (exposure_gain.gain != gain) 
                    {
                        cv::setTrackbarPos("Gain:", "", (int)exposure_gain.gain);

                        std::string action = "Update Gain";
                        std::string log_file_name = "auto_exposure_log.txt"; // Replace with the desired log file name
                        log_changes(log_file_name, action, autoExposureControl.get_current_msv(), autoExposureControl.get_target_msv(), exposure, exposure_gain.gain, noise_level, entropy, sharpness, wbValues.red, wbValues.green, wbValues.blue);
                    }
                    profiler.stop("AutoExposure");
                }
            }        

            if (isBoxSelected)
            {
                auto cropFrame = frameDebayered(fullFrameBox).clone();

                profiler.start("Metrics");
                noise_level = sky360lib::utils::Utils::estimate_noise(cropFrame);
                entropy = sky360lib::utils::Utils::estimate_entropy(cropFrame);
                sharpness = sky360lib::utils::Utils::estimate_sharpness(cropFrame);
                noise_buffer.push_back(noise_level);
                sharpness_buffer.push_back(sharpness);
                profiler.stop("Metrics");

                auto sharpness_graph = sky360lib::utils::Utils::draw_graph("Sharpness", sharpness_buffer, cv::Size(200, 100), CV_8UC3, cv::Scalar(0.0, 255.0, 255.0, 0.0), cv::Scalar(0.0, 25.0, 25.0, 0.0));
                auto noise_graph = sky360lib::utils::Utils::draw_graph("Noise", noise_buffer, cv::Size(200, 100), CV_8UC3, cv::Scalar(0.0, 165.0, 255.0, 0.0), cv::Scalar(0.0, 5.0, 25.0, 0.0));
                sky360lib::utils::Utils::overlay_image(cropFrame, sharpness_graph, cv::Point(0, cropFrame.size().height - sharpness_graph.size().height), 0.7);
                sky360lib::utils::Utils::overlay_image(cropFrame, noise_graph, cv::Point(cropFrame.size().width - sharpness_graph.size().width, cropFrame.size().height - sharpness_graph.size().height), 0.7);

                textWriterCrop.write_text(cropFrame, "Entropy: " + sky360lib::utils::Utils::format_double(entropy), 1, true);

                cv::imshow("Window Cut", cropFrame);
            }
            
            cv::Mat displayFrame;
            if (bgsType != NoBGS)
            {
                bgsPtr->apply(frame, displayFrame);
            }
            else
            {
                displayFrame = frameDebayered;
            }

            drawBoxes(displayFrame);
            if (!squareResolution)
            {
                drawFOV(displayFrame, 220.0, circleCenter, circleRadius);
            }
            else
            {
                drawFOV(displayFrame, cameraCircleMaxFov, cv::Point(frameDebayered.size().width / 2, frameDebayered.size().height / 2), frameDebayered.size().width / 2);
            }
            const double exposure = (double)qhyCamera.get_camera_params().exposure; 
            textWriter.write_text(displayFrame, "Exposure: " + sky360lib::utils::Utils::format_double(exposure / 1000.0, 2) + " ms, Gain: " + std::to_string(qhyCamera.get_camera_params().gain), 1);
            textWriter.write_text(displayFrame, "Resolution: " + std::to_string(qhyCamera.get_camera_params().roi.width) + " x " + std::to_string(qhyCamera.get_camera_params().roi.height) + " (" + std::to_string(qhyCamera.get_camera_params().bpp) + " bits)", 2);
            textWriter.write_text(displayFrame, "Video Recording: " + std::string(isVideoOpen ? "On" : "Off"), 3);
            textWriter.write_text(displayFrame, "Image Equalization: " + std::string(doEqualization ? "On" : "Off"), 4);
            textWriter.write_text(displayFrame, "BGS: " + getBGSName(bgsType), 5);

            textWriter.write_text(displayFrame, "Max Capture FPS: " + sky360lib::utils::Utils::format_double(profileData["GetImage"].fps(), 2), 1, true);
            textWriter.write_text(displayFrame, "Frame FPS: " + sky360lib::utils::Utils::format_double(profileData["Frame"].fps(), 2), 2, true);

            textWriter.write_text(displayFrame, "Auto WB: " + std::string(doAutoWhiteBalance ? "On" : "Off"), 4, true);
            textWriter.write_text(displayFrame, "Auto Exposure: " + std::string(doAutoExposure ? "On" : "Off") + ", Mode: " + (autoExposureControl.is_day() ? "Day" : "Night"), 5, true);
            textWriter.write_text(displayFrame, "MSV: Target " + sky360lib::utils::Utils::format_double(autoExposureControl.get_target_msv()) + ", Current: " + sky360lib::utils::Utils::format_double(autoExposureControl.get_current_msv()), 6, true);
            textWriter.write_text(displayFrame, "Temp.: Cur: " + sky360lib::utils::Utils::format_double(qhyCamera.get_current_temp()) + "c, Targ: " + sky360lib::utils::Utils::format_double(qhyCamera.get_camera_params().target_temp) + "c (" + std::string(qhyCamera.get_camera_params().cool_enabled ? "On" : "Off") + ")", 8, true);

            cv::imshow("Live Video", displayFrame);
            if (showHistogram)
            {
                cv::Mat hist = sky360lib::utils::Utils::create_histogram(frameDebayered);
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
        if (profiler.get_data("Frame").duration_in_seconds() > 1.0)
        {
            profileData = profiler.get_data();
            profiler.reset();
        }
    }
    std::cout << "Exit loop\n"
              << std::endl;

    qhyCamera.close();

    return 0;
}

void createControlPanel()
{
    double aspectRatio = (double)qhyCamera.get_camera_info()->chip.max_image_width / (double)qhyCamera.get_camera_info()->chip.max_image_height;
    cv::namedWindow("Live Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live Video", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_WIDTH / aspectRatio);
    cv::setMouseCallback("Live Video", mouseCallBackFunc, NULL);

    int maxUsbTraffic = (int)qhyCamera.get_camera_info()->usb_traffic_limits.max;
    cv::createTrackbar("USB Traffic:", "", nullptr, maxUsbTraffic, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::UsbTraffic);
    cv::setTrackbarPos("USB Traffic:", "", (int)qhyCamera.get_camera_params().usb_traffic);
    cv::createButton("0.1 ms", exposureCallback, (void *)(long)100, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("1 ms", exposureCallback, (void *)(long)1000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("10 ms", exposureCallback, (void *)(long)10000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("100 ms", exposureCallback, (void *)(long)100000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("1 s", exposureCallback, (void *)(long)1000000, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("- 10%", exposureCallback, (void *)(long)-2, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("+ 10%", exposureCallback, (void *)(long)-1, cv::QT_PUSH_BUTTON, 1);
    int maxGain = (int)qhyCamera.get_camera_info()->gain_limits.max;
    cv::createTrackbar("Gain:", "", nullptr, maxGain, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::Gain);
    cv::setTrackbarPos("Gain:", "", (int)qhyCamera.get_camera_params().gain);
    int maxOffset = (int)qhyCamera.get_camera_info()->offset_limits.max;
    cv::createTrackbar("Offset:", "", nullptr, maxOffset, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::Offset);
    cv::setTrackbarPos("Offset:", "", (int)qhyCamera.get_camera_params().offset);
    cv::createButton("8 bits", TransferbitsCallback, (void *)(long)8, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("16 bits", TransferbitsCallback, (void *)(long)16, cv::QT_PUSH_BUTTON, 1);

    int maxRedWB = (int)qhyCamera.get_camera_info()->red_wb_limits.max;
    cv::createTrackbar("Red WB:", "", nullptr, maxRedWB, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::RedWB);
    cv::setTrackbarPos("Red WB:", "", (int)qhyCamera.get_camera_params().red_white_balance);
    int maxGreenWB = (int)qhyCamera.get_camera_info()->green_wb_limits.max;
    cv::createTrackbar("Green WB:", "", nullptr, maxGreenWB, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::GreenWB);
    cv::setTrackbarPos("Green WB:", "", (int)qhyCamera.get_camera_params().green_white_balance);
    int maxBlueWB = (int)qhyCamera.get_camera_info()->blue_wb_limits.max;
    cv::createTrackbar("Blue WB:", "", nullptr, maxBlueWB, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::BlueWB);
    cv::setTrackbarPos("Blue WB:", "", (int)qhyCamera.get_camera_params().blue_white_balance);

    cv::createButton("Cooling on/off", generalCallback, (void *)(long)'k', cv::QT_PUSH_BUTTON, 1);

    auto temperature_limits = qhyCamera.get_camera_info()->temperature_limits;
    cv::createTrackbar("Temperature Control:", "", nullptr, temperature_limits.max, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::Cooler);
    cv::setTrackbarMin("Temperature Control:", "", (int)temperature_limits.min);
    cv::setTrackbarMax("Temperature Control:", "", (int)temperature_limits.max);
    cv::setTrackbarPos("Temperature Control:", "", (int)qhyCamera.get_camera_params().target_temp);

    cv::createButton("Auto-Exposure on/off", generalCallback, (void *)(long)'a', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("- 10%", exposureCallback, (void *)(long)-4, cv::QT_PUSH_BUTTON, 1);
    cv::createButton("+ 10%", exposureCallback, (void *)(long)-3, cv::QT_PUSH_BUTTON, 1);

    cv::createTrackbar("Auto-Exposure MSV:", "", nullptr, 100.0, changeTrackbars, (void *)(long)-1);
    cv::setTrackbarPos("Auto-Exposure MSV:", "", (int)(autoExposureControl.get_target_msv() * 100.0));

    cv::createButton("Auto WB", generalCallback, (void *)(long)'w', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Square Res.", generalCallback, (void *)(long)'s', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Image Eq.", generalCallback, (void *)(long)'e', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Video Rec.", generalCallback, (void *)(long)'v', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Histogram", generalCallback, (void *)(long)'h', cv::QT_PUSH_BUTTON, 1);
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
            double exposure = (double)qhyCamera.get_camera_params().exposure * 1.1;
            qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure);
            if(doAutoExposure)
            {
                double targetMSV = autoExposureControl.get_target_msv();
                autoExposureControl.set_target_msv(targetMSV * 1.1);
            }
        }
        break;
    case '-':
        {
            double exposure = (double)qhyCamera.get_camera_params().exposure * 0.9;
            qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure);
            if(doAutoExposure)
            {
                double targetMSV = autoExposureControl.get_target_msv();
                autoExposureControl.set_target_msv(targetMSV * 0.9);
            }
        }
        break;
    case '1':
        qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::TransferBits, 8);
        break;
    case '2':
        qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::TransferBits, 16);
        break;
    case 'b':
        bgsType = bgsType == BGSType::WMV ? BGSType::Vibe : (bgsType == BGSType::Vibe ? BGSType::NoBGS : BGSType::WMV);
        bgsPtr = createBGS(bgsType);
        std::cout << "Setting BGS to: " << std::to_string(bgsType) << std::endl;
        break;
    case 's':
        squareResolution = !squareResolution;
        isBoxSelected = false;
        if (squareResolution)
        {
            if (circleSet)
            {
                double max_radius = std::min(std::min(circleCenter.y, qhyCamera.get_camera_info()->chip.max_image_width - circleCenter.y), circleRadius);
                cameraCircleMaxFov = (max_radius / circleRadius) * 220.0;

                uint32_t width = ((uint32_t)max_radius * 2);
                uint32_t height = width;
                uint32_t x = (uint32_t)(circleCenter.x - (width / 2)) & ~0x1;
                uint32_t y = (uint32_t)(circleCenter.y - (height / 2)) & ~0x1;

                qhyCamera.set_resolution(x, y, width, height);
            }
            else
            {
                uint32_t x = ((uint32_t)qhyCamera.get_camera_info()->chip.max_image_width - (uint32_t)qhyCamera.get_camera_info()->chip.max_image_height) / 2;
                uint32_t y = 0;
                uint32_t width = qhyCamera.get_camera_info()->chip.max_image_height;
                uint32_t height = qhyCamera.get_camera_info()->chip.max_image_height;

                qhyCamera.set_resolution(x, y, width, height);
            }
        }
        else
        {
            uint32_t x = 0;
            uint32_t y = 0;
            uint32_t width = qhyCamera.get_camera_info()->chip.max_image_width;
            uint32_t height = qhyCamera.get_camera_info()->chip.max_image_height;
            qhyCamera.set_resolution(x, y, width, height);
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
    case 'k':
        {
            bool is_cool_enabled = qhyCamera.get_camera_params().cool_enabled;
            double target_value = qhyCamera.get_camera_params().target_temp;
            qhyCamera.set_cool_temp(target_value, !is_cool_enabled);
        }
        break;
    case 'w':
        doAutoWhiteBalance = !doAutoWhiteBalance;
        break;
    }
}

void changeTrackbars(int value, void *paramP)
{
    long param = (long)paramP;
    if (param == -1)
    {
        autoExposureControl.set_target_msv((double)value / 100.0);
        return;
    }
    qhyCamera.set_control((sky360lib::camera::QhyCamera::ControlParam)param, (double)value);
}

void exposureCallback(int, void*userData)
{
    double exposure = (double)(long)userData;

    if ((long)userData == -1)
    {
        exposure = (double)qhyCamera.get_camera_params().exposure * 1.1;
    }
    else if ((long)userData == -2)
    {
        exposure = (double)qhyCamera.get_camera_params().exposure * 0.9;
    }
    else if ((long)userData == -3)
    {
        double targetMSV = autoExposureControl.get_target_msv();
        autoExposureControl.set_target_msv(targetMSV * 1.1);
        return;
    }
    else if ((long)userData == -4)
    {
        double targetMSV = autoExposureControl.get_target_msv();
        autoExposureControl.set_target_msv(targetMSV * 0.9);
        return;
    }

    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure);
}

void log_changes(const std::string& log_file_name, const std::string& action, double msv, double targetMSV, double exposure, double gain, double noise_level, double entropy, double sharpness, double redWB, double blueWB, double greenWB)
{
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::ofstream log_file(log_file_name, std::ios_base::app);

    log_file << std::put_time(std::localtime(&time), "%F %T") << ", "
             << action << ", "
             << "msv: " << msv << ", "
             << "targetMSV: " << targetMSV << ", "
             << "exposure: " << exposure << ", "
             << "gain: " << gain << ", "
             << "noise: " << noise_level << ", "
             << "entropy: " << entropy << ", "
             << "sharpness: " << sharpness << ", "
             << "redWB: " << redWB << ", "
             << "blueWB: " << blueWB << ", "
             << "greenWB: " << greenWB << "\n";

    log_file.close();
}

void TransferbitsCallback(int, void*userData)
{
    long transferBits = (long)userData;
    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::TransferBits, transferBits);
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

        std::string fov_text = sky360lib::utils::Utils::format_double(fov, 2);
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
