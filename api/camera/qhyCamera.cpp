#include "qhyCamera.hpp"

#include <chrono>
#include <thread>

namespace sky360lib::camera
{

    std::string QHYCamera::CameraInfo::bayerFormatToString() const
    {
        switch (bayerFormat)
        {
        case BAYER_GB:
            return "BAYER_GB";
        case BAYER_GR:
            return "BAYER_GR";
        case BAYER_BG:
            return "BAYER_BG";
        case BAYER_RG:
            return "BAYER_RG";
        }
        return "MONO";
    }

    std::string QHYCamera::CameraInfo::toString() const
    {
        std::stringstream toStr;
        toStr << "Camera model: " << model << ", Serial: " << serialNum << ", Id: " << id << std::endl;
        toStr << "Overscan  Area startX x startY: " << overscanStartX << " x " << overscanStartY
              << ", sizeX x sizeY : " << overscanWidth << " x " << overscanHeight << std::endl;
        toStr << "Effective Area startX x startY: " << effectiveStartX << " x " << effectiveStartY
              << ", sizeX x sizeY : " << effectiveWidth << " x " << effectiveHeight << std::endl;
        toStr << "Chip      Size width x height: " << chipWidthMM << " x " << chipHeightMM << " [mm]" << std::endl;
        toStr << "Max Image Size width x height: " << maxImageWidth << " x " << maxImageHeight << std::endl;
        toStr << "Pixel     Size width x height: " << pixelWidthUM << " x " << pixelHeightUM << " [um]" << std::endl;
        toStr << "Bits per Pixel: " << bpp << std::endl;
        toStr << "Camera is color: " << (isColor ? "Yes" : "No") << ", Bayer Pattern: " << bayerFormatToString() << std::endl;
        toStr << "Available Bin modes:"
              << (hasBin1x1Mode ? " (1x1)" : "")
              << (hasBin2x2Mode ? " (2x2)" : "")
              << (hasBin3x3Mode ? " (3x3)" : "")
              << (hasBin4x4Mode ? " (4x4)" : "")
              << std::endl;
        toStr << "Gain Limits: Min: " << gainLimits.min << ", Max: " << gainLimits.max << ", Step: " << gainLimits.step << std::endl;
        toStr << "Offset Limits: Min: " << offsetLimits.min << ", Max: " << offsetLimits.max << ", Step: " << offsetLimits.step << std::endl;
        toStr << "Usb Traffic Limits: Min: " << usbTrafficLimits.min << ", Max: " << usbTrafficLimits.max << ", Step: " << usbTrafficLimits.step << std::endl;
        return toStr.str();
    }

    QHYCamera::QHYCamera()
        : m_debugInfo{false}, m_currentInfo{nullptr}
    {
        EnableQHYCCDMessage(false);
        EnableQHYCCDLogFile(false);
    }

    QHYCamera::~QHYCamera()
    {
        release();
    }

    void QHYCamera::setDebugInfo(bool _enable)
    {
        m_debugInfo = _enable;
    }

    bool QHYCamera::getSingle(uint32_t *w, uint32_t *h, uint32_t *bpp, uint32_t *channels, uint8_t *imgData)
    {
        int tries = 0;
        ExpQHYCCDSingleFrame(pCamHandle);
        while (GetQHYCCDSingleFrame(pCamHandle, w, h, bpp, channels, imgData) != QHYCCD_SUCCESS)
        {
            usleep(10000);
            if (++tries > DEFAULT_CAPTURE_RETRIES)
            {
                std::cout << "retries: " << tries << ", aborting." << std::endl;
                return false;
            }
        }
        if (m_debugInfo)
        {
            std::cout << "Gotframe: " << *w << "x" << *h << " pixels, " << *bpp << "bpp, " << *channels << " channels, tries: " << tries << std::endl;
        }
        // CancelQHYCCDExposingAndReadout(pCamHandle);

        return true;
    }

    bool QHYCamera::getLive(uint32_t *w, uint32_t *h, uint32_t *bpp, uint32_t *channels, uint8_t *imgData)
    {
        int tries = 0;
        while (GetQHYCCDLiveFrame(pCamHandle, w, h, bpp, channels, imgData) != QHYCCD_SUCCESS)
        {
            usleep(10000);
            if (++tries > DEFAULT_CAPTURE_RETRIES)
            {
                std::cout << "retries: " << tries << ", aborting." << std::endl;
                return false;
            }
        }
        if (m_debugInfo)
        {
            std::cout << "Gotframe: " << *w << "x" << *h << " pixels, " << *bpp << "bpp, " << *channels << " channels, tries: " << tries << std::endl;
        }
        return true;
    }

    const uint8_t *QHYCamera::getFrame()
    {
        using fsec = std::chrono::duration<double>;
        uint32_t w, h, bpp, channels;

        if (!m_isExposing)
        {
            beginExposing();
        }

        auto start = std::chrono::high_resolution_clock::now();

        if (m_params.streamMode == SingleFrame)
        {
            if (!getSingle(&w, &h, &bpp, &channels, m_pImgData))
            {
                return nullptr;
            }
        }
        else
        {
            if (!getLive(&w, &h, &bpp, &channels, m_pImgData))
            {
                return nullptr;
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        fsec duration = (stop - start);
        m_lastFrameCaptureTime = duration.count();

        return m_pImgData;
    }

    static inline int convertBayerPattern(uint32_t bayerFormat)
    {
        switch (bayerFormat)
        {
        case BAYER_GB:
            return cv::COLOR_BayerGR2BGR; //!< equivalent to GBRG Bayer pattern
        case BAYER_GR:
            return cv::COLOR_BayerGB2BGR; //!< equivalent to GRBG Bayer pattern
        case BAYER_BG:
            return cv::COLOR_BayerRG2BGR; //!< equivalent to BGGR Bayer pattern
        case BAYER_RG:
            return cv::COLOR_BayerBG2BGR; //!< equivalent to RGGB Bayer pattern
        }
        return cv::COLOR_BayerGR2BGR;
    }

    bool QHYCamera::getFrame(cv::Mat &frame, bool debayer)
    {
        const uint8_t *pFrame = getFrame();
        if (!pFrame)
        {
            return false;
        }

        int channels = m_currentInfo->isColor && m_params.applyDebayer ? 3 : 1;
        int type = m_params.transferBits == 16 ? CV_MAKETYPE(CV_16U, channels) : CV_MAKETYPE(CV_8U, channels);

        const cv::Mat imgQHY(m_params.roiHeight, m_params.roiWidth, type, (int8_t *)pFrame);

        if (m_currentInfo->isColor && !m_params.applyDebayer && debayer)
        {
            debayerImage(imgQHY, frame);
        }
        else
        {
            imgQHY.copyTo(frame);
        }

        return true;
    }

    void QHYCamera::debayerImage(const cv::Mat &imageIn, cv::Mat &imageOut) const
    {
        if (imageIn.channels() == 1)
        {
            cv::cvtColor(imageIn, imageOut, convertBayerPattern(m_currentInfo->bayerFormat));
        }
        else
        {
            imageIn.copyTo(imageOut);
        }
    }

    cv::Mat QHYCamera::getFrameRet(bool debayer)
    {
        cv::Mat returnFrame;
        getFrame(returnFrame, debayer);
        return returnFrame;
    }

    bool QHYCamera::fillCameraInfo(std::string camId, CameraInfo &ci)
    {
        qhyccd_handle *camHandle = OpenQHYCCD((char *)camId.c_str());
        if (camHandle == nullptr)
        {
            std::cerr << "OpenQHYCCD failure, camera id: " << camId << std::endl;
            return false;
        }

        ci.id = camId;
        size_t posDash = ci.id.find("-");
        ci.model = ci.id.substr(0, posDash);
        ci.serialNum = ci.id.substr(posDash + 1);

        uint32_t rc = GetQHYCCDOverScanArea(camHandle, &ci.overscanStartX, &ci.overscanStartY, &ci.overscanWidth, &ci.overscanHeight);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "GetQHYCCDOverScanArea failure, camera id: " << camId << ", error: " << rc << std::endl;
            return false;
        }

        rc = GetQHYCCDEffectiveArea(camHandle, &ci.effectiveStartX, &ci.effectiveStartY, &ci.effectiveWidth, &ci.effectiveHeight);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "GetQHYCCDEffectiveArea failure, camera id: " << camId << ", error: " << rc << std::endl;
            return false;
        }

        rc = GetQHYCCDChipInfo(camHandle, &ci.chipWidthMM, &ci.chipHeightMM, &ci.maxImageWidth, &ci.maxImageHeight,
                               &ci.pixelWidthUM, &ci.pixelHeightUM, &ci.bpp);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "GetQHYCCDChipInfo failure," << std::endl;
            return false;
        }

        ci.bayerFormat = IsQHYCCDControlAvailable(camHandle, CAM_COLOR);
        ci.isColor = (ci.bayerFormat == BAYER_GB || ci.bayerFormat == BAYER_GR || ci.bayerFormat == BAYER_BG || ci.bayerFormat == BAYER_RG);

        ci.hasBin1x1Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN1X1MODE) == QHYCCD_SUCCESS;
        ci.hasBin2x2Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN2X2MODE) == QHYCCD_SUCCESS;
        ci.hasBin3x3Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN3X3MODE) == QHYCCD_SUCCESS;
        ci.hasBin4x4Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN4X4MODE) == QHYCCD_SUCCESS;

        GetQHYCCDParamMinMaxStep(camHandle, CONTROL_GAIN, &ci.gainLimits.min, &ci.gainLimits.max, &ci.gainLimits.step);
        GetQHYCCDParamMinMaxStep(camHandle, CONTROL_OFFSET, &ci.offsetLimits.min, &ci.offsetLimits.max, &ci.offsetLimits.step);
        GetQHYCCDParamMinMaxStep(camHandle, CONTROL_USBTRAFFIC, &ci.usbTrafficLimits.min, &ci.usbTrafficLimits.max, &ci.usbTrafficLimits.step);
        GetQHYCCDParamMinMaxStep(camHandle, CONTROL_WBR, &ci.redWBLimits.min, &ci.redWBLimits.max, &ci.redWBLimits.step);
        GetQHYCCDParamMinMaxStep(camHandle, CONTROL_WBG, &ci.greenWBLimits.min, &ci.greenWBLimits.max, &ci.greenWBLimits.step);
        GetQHYCCDParamMinMaxStep(camHandle, CONTROL_WBB, &ci.blueWBLimits.min, &ci.blueWBLimits.max, &ci.blueWBLimits.step);

        rc = CloseQHYCCD(camHandle);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "Close QHYCCD failure, error: " << rc << std::endl;
        }
        if (m_debugInfo)
        {
            std::cout << ci.toString() << std::endl;
        }

        return true;
    }

    bool QHYCamera::scanCameras()
    {
        init();

        m_cameras.clear();

        const int camCount = ScanQHYCCD();
        if (camCount <= 0)
        {
            std::cerr << "No QHYCCD camera found, please check USB or power." << std::endl;
            return false;
        }

        char camId[64];
        for (int i{0}; i < camCount; ++i)
        {
            uint32_t rc = GetQHYCCDId(i, camId);
            if (rc == QHYCCD_SUCCESS)
            {
                CameraInfo ci;
                if (fillCameraInfo(camId, ci))
                {
                    m_cameras[camId] = ci;
                }
            }
        }

        if (m_cameras.size() == 0)
        {
            release();
            return false;
        }

        return true;
    }

    const std::map<std::string, QHYCamera::CameraInfo> &QHYCamera::getCameras()
    {
        scanCameras();
        return m_cameras;
    }

    QHYCamera::CameraInfo const *QHYCamera::getCameraInfo() const
    {
        return m_currentInfo;
    }

    const QHYCamera::CameraParams &QHYCamera::getCameraParams() const
    {
        return m_params;
    }

    double QHYCamera::getLastFrameCaptureTime() const
    {
        return m_lastFrameCaptureTime;
    }

    bool QHYCamera::setControl(ControlParam controlParam, double value, bool force)
    {
        uint32_t rc = IsQHYCCDControlAvailable(pCamHandle, (CONTROL_ID)controlParam);
        if (rc == QHYCCD_SUCCESS)
        {
            switch (controlParam)
            {
            case Channels:
                if (m_params.channels == (uint32_t)value)
                    return true;
                m_params.channels = value;
                if (m_camOpen)
                {
                    allocBufferMemory();
                    beginExposing();
                }
                break;
            case TransferBits:
                if (!force && m_params.transferBits == (uint32_t)value)
                    return true;
                m_params.transferBits = value;
                if (m_camOpen)
                {
                    allocBufferMemory();
                    close();
                    open(m_camId);
                }
                break;
            case Brightness:
                if (!force && m_params.brightness == value)
                    return true;
                m_params.brightness = value;
                break;
            case Contrast:
                if (!force && m_params.contrast == value)
                    return true;
                m_params.contrast = value;
                break;
            case Exposure:
                if (!force && m_params.exposureTime == (uint32_t)value)
                    return true;
                m_params.exposureTime = value;
                break;
            case UsbTraffic:
                if (!force && m_params.usbTraffic == (uint32_t)value)
                    return true;
                m_params.usbTraffic = value;
                break;
            case UsbSpeed:
                if (!force && m_params.usbSpeed == (uint32_t)value)
                    return true;
                m_params.usbSpeed = value;
                break;
            case Gain:
                if (!force && m_params.gain == (uint32_t)value)
                    return true;
                m_params.gain = value;
                break;
            case Offset:
                if (!force && m_params.offset == (uint32_t)value)
                    return true;
                m_params.offset = value;
                break;
            case RedWB:
                if (!force && m_params.redWB == value)
                    return true;
                m_params.redWB = value;
                break;
            case GreenWB:
                if (!force && m_params.greenWB == value)
                    return true;
                m_params.greenWB = value;
                break;
            case BlueWB:
                if (!force && m_params.blueWB == value)
                    return true;
                m_params.blueWB = value;
                break;
            case Gamma:
                if (!force && m_params.gamma == value)
                    return true;
                m_params.gamma = value;
                break;
            default:
                break;
            }
            rc = SetQHYCCDParam(pCamHandle, (CONTROL_ID)controlParam, value);
            if (rc != QHYCCD_SUCCESS)
            {
                std::cerr << "setControl failed: " << controlParam << std::endl;
                return false;
            }

        }
        else if (m_debugInfo)
        {
            std::cout << "Control not available to change: " << controlParam << std::endl;
        }
        return true;
    }

    bool QHYCamera::setDebayer(bool enable)
    {
        if (SetQHYCCDDebayerOnOff(pCamHandle, enable) != QHYCCD_SUCCESS)
        {
            std::cerr << "SetQHYCCDDebayerOnOff failure" << std::endl;
            return false;
        }
        allocBufferMemory();
        m_params.applyDebayer = enable;

        return true;
    }

    bool QHYCamera::setBinMode(QHYCamera::BinMode mode)
    {
        uint32_t rc = SetQHYCCDBinMode(pCamHandle, (int)mode, (int)mode);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "SetQHYCCDBinMode failure, error: " << rc << std::endl;
            return false;
        }
        allocBufferMemory();
        m_params.binMode = mode;

        return true;
    }

    bool QHYCamera::setResolution(uint32_t startX, uint32_t startY, uint32_t width, uint32_t height)
    {
        uint32_t rc = SetQHYCCDResolution(pCamHandle, startX, startY, width, height);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "SetQHYCCDResolution failure, error: " << rc << std::endl;
            return false;
        }
        allocBufferMemory();
        m_params.roiStartX = startX;
        m_params.roiStartY = startY;
        m_params.roiWidth = width;
        m_params.roiHeight = height;

        return true;
    }

    bool QHYCamera::setStreamMode(StreamModeType mode)
    {
        uint32_t rc = SetQHYCCDStreamMode(pCamHandle, (int)mode);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "Error setting stream mode" << std::endl;
            return false;
        }
        m_params.streamMode = mode;

        rc = InitQHYCCD(pCamHandle);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "InitQHYCCD faililure" << std::endl;
            return false;
        }
        // applyParams();

        return true;
    }

    uint32_t QHYCamera::getMemoryNeededForFrame() const
    {
        return GetQHYCCDMemLength(pCamHandle);
    }

    bool QHYCamera::init()
    {
        if (!m_camInit)
        {
            if (InitQHYCCDResource() != QHYCCD_SUCCESS)
            {
                std::cerr << "Cannot initialize SDK resources" << std::endl;
                m_camInit = false;
                return false;
            }
            m_camInit = true;
        }

        return true;
    }

    void QHYCamera::setDefaultParams()
    {
        if (!m_defaultSet)
        {
            setDebayer(false);
            setControl(RedWB, 180.0, true);
            setControl(GreenWB, 128.0, true);
            setControl(BlueWB, 190.0, true);
            setControl(Exposure, 2000, true);
            setStreamMode(LiveFrame);
            setControl(UsbTraffic, 0, true);
            setControl(UsbSpeed, 0, true);
            setControl(Gain, 30, true);
            setControl(Offset, 0, true);
            setResolution(0, 0, getCameraInfo()->maxImageWidth, getCameraInfo()->maxImageHeight);
            setControl(TransferBits, 16, true);
            setControl(Channels, 1, true);
            setBinMode(Bin_1x1);
            setControl(Contrast, 0.0, true);
            setControl(Brightness, 0.0, true);
            setControl(Gamma, 1.0, true);

            m_defaultSet = true;
        }
        else
        {
            applyParams();
        }
    }

    void QHYCamera::applyParams()
    {
        setDebayer(m_params.applyDebayer);
        setControl(RedWB, m_params.redWB);
        setControl(GreenWB, m_params.greenWB);
        setControl(BlueWB, m_params.blueWB);
        setControl(Exposure, m_params.exposureTime);
        setStreamMode(m_params.streamMode);
        setControl(UsbTraffic, m_params.usbTraffic);
        setControl(UsbSpeed, m_params.usbSpeed);
        setControl(Gain, m_params.gain);
        setControl(Offset, m_params.offset);
        setResolution(m_params.roiStartX, m_params.roiStartY, m_params.roiWidth, m_params.roiHeight);
        setControl(TransferBits, m_params.transferBits);
        setControl(Channels, m_params.channels);
        setBinMode(m_params.binMode);
        setControl(Contrast, m_params.contrast);
        setControl(Brightness, m_params.brightness);
        setControl(Gamma, m_params.gamma);
    }

    void QHYCamera::release()
    {
        if (m_camOpen)
        {
            close();
        }

        // release sdk resources
        uint32_t rc = ReleaseQHYCCDResource();
        if (QHYCCD_SUCCESS != rc)
        {
            std::cerr << "Cannot release SDK resources, error: " << rc << std::endl;
        }
    }

    bool QHYCamera::allocBufferMemory()
    {
        if (m_pImgData != nullptr)
        {
            releaseBufferMemory();
        }
        uint32_t size = getMemoryNeededForFrame();
        if (size == 0)
        {
            std::cerr << "Cannot get memory for frame." << std::endl;
            return false;
        }

        m_pImgData = new uint8_t[size];
        memset(m_pImgData, 0, size);

        return true;
    }

    void QHYCamera::releaseBufferMemory()
    {
        delete[] m_pImgData;
        m_pImgData = nullptr;
    }

    void QHYCamera::endExposing()
    {
        if (m_params.streamMode == SingleFrame)
        {
            CancelQHYCCDExposingAndReadout(pCamHandle);
        }
        else
        {
            StopQHYCCDLive(pCamHandle);
        }
        m_isExposing = false;
    }

    bool QHYCamera::beginExposing()
    {
        if (m_params.streamMode == SingleFrame)
        {
            CancelQHYCCDExposingAndReadout(pCamHandle);
            uint32_t rc = ExpQHYCCDSingleFrame(pCamHandle);
            if (rc != QHYCCD_ERROR)
            {
                // std::cout << "ExpQHYCCDSingleFrame returned: " << rc << std::endl;
                if (rc == QHYCCD_READ_DIRECTLY)
                {
                    usleep(1);
                }
            }
            else
            {
                std::cerr << "ExpQHYCCDSingleFrame failed: " << rc << std::endl;
                return false;
            }
        }
        else
        {
            StopQHYCCDLive(pCamHandle);
            uint32_t rc = BeginQHYCCDLive(pCamHandle);
            if (rc != QHYCCD_SUCCESS)
            {
                std::cerr << "ExpQHYCCDSingleFrame failed: " << rc << std::endl;
                return false;
            }
        }
        m_isExposing = true;
        return true;
    }

    bool QHYCamera::open(std::string cameraId)
    {
        if (!m_camInit && !init())
        {
            return false;
        }
        if (!m_camOpen)
        {
            if (cameraId.empty())
            {
                if (!scanCameras())
                {
                    return false;
                }
                cameraId = m_cameras.begin()->second.id;
            }
            else if (cameraId != m_camId)
            {
                m_defaultSet = false;
            }
            m_camId = cameraId;
            m_currentInfo = &m_cameras[m_camId];

            pCamHandle = OpenQHYCCD((char *)m_camId.c_str());
            if (pCamHandle == nullptr)
            {
                m_camId = "";
                m_currentInfo = nullptr;
                std::cerr << "Open QHYCCD failure." << std::endl;
                return false;
            }

            // uint32_t rc = InitQHYCCD(pCamHandle);
            // if (rc != QHYCCD_SUCCESS)
            // {
            //     std::cerr << "InitQHYCCD faililure" << std::endl;
            //     return false;
            // }

            setDefaultParams();
            m_camOpen = true;
        }

        return m_camOpen;
    }

    void QHYCamera::close()
    {
        if (m_camOpen)
        {
            if (m_params.streamMode == SingleFrame)
            {
                CancelQHYCCDExposingAndReadout(pCamHandle);
            }
            else
            {
                StopQHYCCDLive(pCamHandle);
                SetQHYCCDStreamMode(pCamHandle, 0x0);
            }

            CloseQHYCCD(pCamHandle);

            releaseBufferMemory();

            pCamHandle = nullptr;
            m_camId = "";
            m_camOpen = false;
            m_isExposing = false;
        }
    }

}