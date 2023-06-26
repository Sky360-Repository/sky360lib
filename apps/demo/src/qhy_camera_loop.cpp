#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "../../../api/camera/qhy_camera.hpp"
#include "../../../api/utils/utils.hpp"
#include "../../../api/utils/autoExposureControl.hpp"
#include "../../../api/utils/autoWhiteBalance.hpp"
#include "../../../api/utils/profiler.hpp"
#include "../../../api/utils/textWriter.hpp"
#include "../../../api/bgs/bgs.hpp"
#include "../../../api/blobs/connectedBlobDetection.hpp"
#include "../../../api/utils/ringbuf.h"
#include "../../../api/utils/roi_mask_calculator.hpp"
#include "../../../api/utils/bin_image.hpp"
#include "../../../api/utils/image_stacker.hpp"

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
bool doAutoExposure = true;
bool doAutoWhiteBalance = false;
bool squareResolution = false;
bool updateDisplayOverlay = false;
bool logData = false;
bool run = true;
bool pauseCapture = false;
bool showHistogram = false;
bool settingCircle = false;
bool circleSet = false;
bool doSoftwareBin = false;
bool doStacking = false;
bool doBlobDetection = false;
BGSType bgsType{WMV};

cv::Rect fullFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Rect tempFrameBox{0, 0, DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE};
cv::Point2d circleInit;
cv::Point2d circleEnd;
cv::Point2d circleCenter;
double circleRadius{0.0f};
double cameraCircleMaxFov{0.0};
baudvine::RingBuf<double, 200> noise_buffer;
baudvine::RingBuf<double, 200> sharpness_buffer;
cv::Mat displayFrame;
std::string home_directory;

cv::VideoWriter videoWriter;
sky360lib::utils::DataMap profileData;
sky360lib::utils::Profiler profiler;
sky360lib::camera::QhyCamera qhyCamera;
sky360lib::utils::TextWriter textWriter(cv::Scalar{190, 190, 190, 0}, 36, 2.0);
sky360lib::utils::AutoExposureControl autoExposureControl;
sky360lib::utils::AutoWhiteBalance auto_white_balance(50000.0);
sky360lib::utils::BinImage bin_image;
sky360lib::utils::ImageStacker image_stacker;
sky360lib::blobs::ConnectedBlobDetection blob_detection;

std::unique_ptr<sky360lib::bgs::CoreBgs> bgsPtr{nullptr};

/////////////////////////////////////////////////////////////
// Function Definitions
inline void drawBoxes(const cv::Mat &frame);
bool openQQYCamera();
bool openVideo(double meanFps);
void createControlPanel();
void treatKeyboardpress(int key);
void changeTrackbars(int value, void *paramP);
void mouseCallBackFunc(int event, int x, int y, int, void *);
void exposureCallback(int, void*userData);
void TransferbitsCallback(int, void*userData);
void generalCallback(int, void*userData);
void drawFOV(cv::Mat& frame, double max_fov, cv::Point2d center, double radius);
std::streampos get_file_size(const std::string& file_name);
void log_changes(const std::string& log_file_name, double msv, double targetMSV, double exposure, double gain, double noise_level, double entropy, double sharpness, double redWB, double blueWB, double greenWB, std::streampos max_file_size);
std::unique_ptr<sky360lib::bgs::CoreBgs> createBGS(BGSType _type);
std::string getBGSName(BGSType _type);
std::string get_running_time(std::chrono::system_clock::time_point input_time);
std::string generate_filename();
inline void drawBboxes(std::vector<cv::Rect> &bboxes, const cv::Mat &frame);

/////////////////////////////////////////////////////////////
// Main entry point for demo
int main(int argc, const char **argv)
{
    // Get the current time
    auto starting_time = std::chrono::system_clock::now();

    home_directory = getenv("HOME");

    if (!openQQYCamera())
    {
        return -1;
    }
    std::cout << qhyCamera.get_camera_info()->to_string() << std::endl;
    qhyCamera.set_debug_info(false);

    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, (argc > 1 ? atoi(argv[1]) : 10000));
    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Gain, 0.0);

    bgsPtr = createBGS(bgsType);

    int frame_counter = 0;
    const int auto_exposure_frame_interval = 3; 
    const int log_interval = 10;

    createControlPanel();

    cv::Mat frame, processedFrame, saveFrame, frameDebayered;

    qhyCamera.get_frame(frame, false);
    frameSize = frame.size();

    double noise_level = 0.0;
    double entropy = 0.0;
    double sharpness = 0.0;

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
            if (doStacking)
            {
                cv::Mat stacked_image;
                image_stacker.stack(frame, stacked_image);
                frame = stacked_image;
            }
            profiler.start("Debayer");
            if (doSoftwareBin)
            {
                cv::Mat binned_image;
                bin_image.apply_bin_2x2(frame, binned_image);
                frameDebayered = binned_image;
                frame = binned_image;
            }
            else
            {
                qhyCamera.debayer_image(frame, frameDebayered);
            }
            frameSize = frameDebayered.size();
            profiler.stop("Debayer");
            if (doEqualization)
            {
                profiler.start("Equalization");
                sky360lib::utils::Utils::equalize_image(frameDebayered, frameDebayered, clipLimit);
                profiler.stop("Equalization");
            }
            if (doAutoWhiteBalance)
            {
                const double current_exposure = qhyCamera.get_camera_params().exposure;
                auto wb_values = auto_white_balance.gray_world(frameDebayered, current_exposure);
                if (wb_values.apply)
                {
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::RedWB, wb_values.red);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::GreenWB, wb_values.green);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::BlueWB, wb_values.blue);

                    cv::setTrackbarPos("Red WB:", "", (int)wb_values.red);
                    cv::setTrackbarPos("Green WB:", "", (int)wb_values.green);
                    cv::setTrackbarPos("Blue WB:", "", (int)wb_values.blue);
                }
            }

            if (doAutoExposure)
            {
                frame_counter++;

                if (frame_counter % auto_exposure_frame_interval == 0) // to improve fps
                { 
                    profiler.start("AutoExposure");
                    const double exposure = (double)qhyCamera.get_camera_params().exposure;
                    const double gain = (double)qhyCamera.get_camera_params().gain;
                    auto exposure_gain = autoExposureControl.calculate_exposure_gain(frame, exposure, gain);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Exposure, exposure_gain.exposure);
                    qhyCamera.set_control(sky360lib::camera::QhyCamera::ControlParam::Gain, exposure_gain.gain);                    

                    // Log gain update
                    if (exposure_gain.gain != gain) 
                    {
                        cv::setTrackbarPos("Gain:", "", (int)exposure_gain.gain);
                    }
                    profiler.stop("AutoExposure");
                }
            }        

            if (isBoxSelected)
            {
                auto cropFrame = frameDebayered(fullFrameBox).clone();

                profiler.start("Metrics");
                noise_level = sky360lib::utils::Utils::estimate_noise(cropFrame);
                sharpness = sky360lib::utils::Utils::estimate_sharpness(cropFrame);
                noise_buffer.push_back(noise_level);
                sharpness_buffer.push_back(sharpness);
                profiler.stop("Metrics");

                auto sharpness_graph = sky360lib::utils::Utils::draw_graph("Sharpness", sharpness_buffer, cv::Size(200, 100), cropFrame.type(), cv::Scalar(0.0, 255.0, 255.0, 0.0), cv::Scalar(0.0, 25.0, 25.0, 0.0));
                auto noise_graph = sky360lib::utils::Utils::draw_graph("Noise", noise_buffer, cv::Size(200, 100), cropFrame.type(), cv::Scalar(0.0, 165.0, 255.0, 0.0), cv::Scalar(0.0, 5.0, 25.0, 0.0));
                sky360lib::utils::Utils::overlay_image(cropFrame, sharpness_graph, cv::Point(0, cropFrame.size().height - sharpness_graph.size().height), 0.7);
                sky360lib::utils::Utils::overlay_image(cropFrame, noise_graph, cv::Point(cropFrame.size().width - noise_graph.size().width, cropFrame.size().height - noise_graph.size().height), 0.7);

                cv::imshow("Window Cut", cropFrame);
            }
            
            if (logData)
            {
                profiler.start("Log Data");
                if (frame_counter % log_interval == 0)
                {
                    entropy = sky360lib::utils::Utils::estimate_entropy(frame);

                    log_changes("log_camera_params.txt", 
                        autoExposureControl.get_current_msv(), 
                        autoExposureControl.get_target_msv(), 
                        qhyCamera.get_camera_params().exposure,
                        qhyCamera.get_camera_params().gain, 
                        noise_level, entropy, sharpness, 
                        qhyCamera.get_camera_params().red_white_balance, 
                        qhyCamera.get_camera_params().blue_white_balance, 
                        qhyCamera.get_camera_params().green_white_balance, 
                        67108864);
                }
                profiler.stop("Log Data");
            }

            displayFrame = frameDebayered;
            
            profiler.start("BGS/Blob");
            if (doBlobDetection)
            {
                cv::Mat bgs_mask;
                bgsPtr->apply(frame, bgs_mask);
                std::vector<cv::Rect> bboxes;
                blob_detection.detect(bgs_mask, bboxes);
                drawBboxes(bboxes, displayFrame);
            }
            profiler.stop("BGS/Blob");

            profiler.start("Display Frame");
            if (!squareResolution)
            {
                drawFOV(displayFrame, 220.0, circleCenter, circleRadius);
            }
            else
            {
                drawFOV(displayFrame, cameraCircleMaxFov, cv::Point(frameDebayered.size().width / 2, frameDebayered.size().height / 2), frameDebayered.size().width / 2);
            }
            int bin_mode = doSoftwareBin ? 2 : qhyCamera.get_camera_params().bin_mode;
            textWriter.write_text(displayFrame, "Exp: " + sky360lib::utils::Utils::format_double((double)qhyCamera.get_camera_params().exposure / 1000.0, 2) + " ms Gain: " + std::to_string(qhyCamera.get_camera_params().gain), 1);
            textWriter.write_text(displayFrame, 
                std::to_string(frame.size().width) + "x" + std::to_string(frame.size().height) + " (" + std::to_string(qhyCamera.get_camera_params().bpp) + " bits " + std::to_string(bin_mode) + "x" + std::to_string(bin_mode) + ")", 2);
            textWriter.write_text(displayFrame, "Camera FPS: " + sky360lib::utils::Utils::format_double(profileData["GetImage"].fps(), 2), 1, true);
            textWriter.write_text(displayFrame, "Frame FPS: " + sky360lib::utils::Utils::format_double(profileData["Frame"].fps(), 2), 2, true);
            textWriter.write_text(displayFrame, get_running_time(starting_time), 36, true);
            if (qhyCamera.get_camera_info()->is_cool)
            {
                textWriter.write_text(displayFrame, sky360lib::utils::Utils::format_double(qhyCamera.get_current_temp()) + "c", 36);
            }
            // textWriter.write_text(displayFrame, "BGS: " + getBGSName(bgsType), 5);
            // textWriter.write_text(displayFrame, "MSV: Target " + sky360lib::utils::Utils::format_double(autoExposureControl.get_target_msv()) + ", Current: " + sky360lib::utils::Utils::format_double(autoExposureControl.get_current_msv()), 6, true);

            std::string features_enabled;
            features_enabled += doAutoWhiteBalance ? "Auto WB: On | " : "";
            features_enabled += isVideoOpen ? "Video Rec. | " : "";
            features_enabled += doEqualization ? "Equalization: On | " : "";
            features_enabled += logData ? "Logging: On | " : "";
            features_enabled += qhyCamera.get_camera_params().cool_enabled ? "Cooling: " + sky360lib::utils::Utils::format_double(qhyCamera.get_camera_params().target_temp) +  " C " : "";
            features_enabled += doAutoExposure ? std::string("Auto Exp: ") + (autoExposureControl.is_day() ? "Day" : "Night") + " | " : "";
            features_enabled += doStacking ? "Stacking: On | " : "";
            features_enabled += bgsType != NoBGS ? ("BGS: " + getBGSName(bgsType) + " | ") : "";
            features_enabled = !features_enabled.empty() ? features_enabled : "No features activated";
            if (updateDisplayOverlay)
            {
                cv::displayOverlay("Live Video", features_enabled, 1500);
                updateDisplayOverlay = false;
            }
            cv::displayStatusBar("Live Video", features_enabled, 0);

            // if (circleSet)
            // {
            //     auto roi_mask = sky360lib::utils::RoiMaskCalculator::calc_roi_mask(displayFrame, circleInit, circleEnd, 180.0, 220.0);
            //     cv::rectangle(displayFrame, cv::Rect(roi_mask.x, roi_mask.y, roi_mask.width, roi_mask.height), cv::Scalar(255, 255, 255), 5);
            //     cv::Mat result;
            //     cv::bitwise_and(displayFrame, displayFrame, result, roi_mask.mask);
            //     cv::imshow("Live Video", result);
            // }
            // else
                cv::Mat live_frame = displayFrame;
                drawBoxes(live_frame);
                cv::imshow("Live Video", live_frame);
            profiler.stop("Display Frame");
            if (showHistogram)
            {
                profiler.start("Display Histogram");
                cv::Mat hist = sky360lib::utils::Utils::create_histogram(frameDebayered);
                cv::imshow("Histogram", hist);
                profiler.stop("Display Histogram");
            }
            if (isVideoOpen)
            {
                profiler.start("Save Video");
                videoWriter.write(displayFrame);
                profiler.stop("Save Video");
            }
        }

        profiler.start("Key Handle");
        treatKeyboardpress(cv::pollKey()); 
        profiler.stop("Key Handle");

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
    profiler.report();

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

    int maxGamma = (int)qhyCamera.get_camera_info()->gamma_limits.max * 10;
    cv::createTrackbar("Gamma:", "", nullptr, maxGamma, changeTrackbars, (void *)(long)sky360lib::camera::QhyCamera::ControlParam::Gamma);
    cv::setTrackbarPos("Gamma:", "", (int)qhyCamera.get_camera_params().gamma * 10);
    cv::createButton("Screenshot", generalCallback, (void *)(long)'i', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Video Rec.", generalCallback, (void *)(long)'v', cv::QT_PUSH_BUTTON, 1);
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

    cv::createButton("Stacking", generalCallback, (void *)(long)'z', cv::QT_PUSH_BUTTON, 1);
    cv::createTrackbar("Stacking:", "", nullptr, 100, changeTrackbars, (void *)(long)-2);
    cv::setTrackbarPos("Stacking:", "", (int)(image_stacker.get_weight() * 100.0));

    cv::createButton("Auto WB", generalCallback, (void *)(long)'w', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Binning", generalCallback, (void *)(long)'n', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Hist Eq.", generalCallback, (void *)(long)'e', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Histogram", generalCallback, (void *)(long)'h', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Square Res.", generalCallback, (void *)(long)'s', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Log data", generalCallback, (void *)(long)'l', cv::QT_PUSH_BUTTON, 1);
    cv::createButton("Exit Program", generalCallback, (void *)(long)27, cv::QT_PUSH_BUTTON, 1);
}

void treatKeyboardpress(int key)
{
    if (!cv::getWindowProperty("Live Video", cv::WND_PROP_VISIBLE))
    {
        run = false;
        return;
    }

    if (key < 0)
        return;

    switch ((char)key)
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
        updateDisplayOverlay = true;
        break;
    case 'v':
        if (!isVideoOpen)
        {
            std::cout << "Start recording" << std::endl;
            isVideoOpen = openVideo(profileData["Frame"].fps());
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
    case 'g':
        bgsType = bgsType == BGSType::WMV ? BGSType::Vibe : BGSType::WMV;
        bgsPtr = createBGS(bgsType);
        std::cout << "Setting BGS to: " << std::to_string(bgsType) << std::endl;
        break;
    case 'b':
        doBlobDetection = !doBlobDetection;
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
    case 'l':
        logData = !logData;
        updateDisplayOverlay = true;
        break;
    case 'a':
        doAutoExposure = !doAutoExposure;
        updateDisplayOverlay = true;
        break;
    case 'k':
        {
            bool is_cool_enabled = qhyCamera.get_camera_params().cool_enabled;
            double target_value = qhyCamera.get_camera_params().target_temp;
            qhyCamera.set_cool_temp(target_value, !is_cool_enabled);
        }
        updateDisplayOverlay = true;
        break;
    case 'w':
        doAutoWhiteBalance = !doAutoWhiteBalance;
        updateDisplayOverlay = true;
        break;
    case 'i':
        {
            std::string filename = home_directory + "/ss_" + generate_filename() + ".png";
            std::cout << "Saving screenshot to: " << filename << std::endl;
            cv::imwrite(filename, displayFrame, {cv::IMWRITE_PNG_COMPRESSION, 5});
        }
        break;
    case 'm':
        {
            if (isBoxSelected)
            {
                isBoxSelected = false;
                cv::destroyWindow("Window Cut");
            }
            auto current_bin = qhyCamera.get_camera_params().bin_mode;
            auto new_bin = current_bin == sky360lib::camera::QhyCamera::Bin1x1 ? sky360lib::camera::QhyCamera::Bin2x2 : sky360lib::camera::QhyCamera::Bin1x1;
            qhyCamera.set_bin_mode(new_bin);
        }
        break;
    case 'n':
        if (isBoxSelected)
        {
            isBoxSelected = false;
            cv::destroyWindow("Window Cut");
        }
        doSoftwareBin = !doSoftwareBin;
        break;
    case 'z':
        doStacking = !doStacking;
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
    else if (param == -2)
    {
        image_stacker.set_weight((double)value / 100.0);
        return;
    }
    else if ((sky360lib::camera::QhyCamera::ControlParam)param == sky360lib::camera::QhyCamera::ControlParam::Gamma)
    {
        qhyCamera.set_control((sky360lib::camera::QhyCamera::ControlParam)param, (double)value / 10.0);
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

std::streampos get_file_size(const std::string& file_name)
{
    std::ifstream file(file_name, std::ifstream::ate | std::ifstream::binary);
    return file.tellg();
}

void log_changes(const std::string& log_file_name, double msv, double targetMSV, double exposure, double gain, double noise_level, double entropy, double sharpness, double redWB, double blueWB, double greenWB, std::streampos max_file_size)
{
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::streampos current_file_size = get_file_size(log_file_name);

    // If the file size exceeds the maximum, truncate the file
    if (current_file_size >= max_file_size) {
        std::ofstream log_file(log_file_name, std::ofstream::trunc);
        log_file.close();
    }

    // Append the log entry
    std::ofstream log_file(log_file_name, std::ios_base::app);

    log_file << std::put_time(std::localtime(&time), "%F %T") << ", "
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
    treatKeyboardpress((int)param);
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
    }
}

void mouseCallBackFunc(int event, int x, int y, int flags, void *)
{
    switch (event)
    {
        case cv::EVENT_MBUTTONUP:
            settingCircle = false;
            circleSet = false;
            fullFrameBox = tempFrameBox;
            if (isBoxSelected)
            {
                cv::destroyWindow("Window Cut");
            }
            isBoxSelected = false;
            break;
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
                noise_buffer.clear();
                sharpness_buffer.clear();
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

std::string generate_filename()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return oss.str();
}

bool openVideo(double meanFps)
{
    auto name = home_directory + "/vo" + generate_filename() + ".mkv";
    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    return videoWriter.open(name, codec, meanFps, displayFrame.size(), displayFrame.channels() == 3);
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

std::string get_running_time(std::chrono::system_clock::time_point input_time) 
{
    // Get the current time
    auto current_time = std::chrono::system_clock::now();

    // Calculate the time difference
    std::chrono::duration<double> diff = current_time - input_time;

    auto h = std::chrono::duration_cast<std::chrono::hours>(diff);
    diff -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(diff);
    diff -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(diff);

    // Convert the time difference into a string in the format HH:MM:SS
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << h.count() << ":"
       << std::setw(2) << std::setfill('0') << m.count() << ":"
       << std::setw(2) << std::setfill('0') << s.count();

    return ss.str();
}

inline void drawBboxes(std::vector<cv::Rect> &bboxes, const cv::Mat &frame)
{
    auto color = frame.channels() == 1 ? cv::Scalar(255, 255, 255) : cv::Scalar(255, 0, 255);
    for (auto bb : bboxes)
    {
        if (frame.elemSize1() == 1)
        {
            cv::rectangle(frame, bb, color, 2);
        }
        else
        {
            cv::rectangle(frame, bb, cv::Scalar(color[0] * 255, color[1] * 255, color[2] * 255), 2);
        }
    }
}