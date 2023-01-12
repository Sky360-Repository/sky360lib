#include "qhyCamera.hpp"

#include <chrono>
#include <thread>

using namespace sky360lib::camera;

std::string QHYCamera::CameraInfo::bayerFormatToString()
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

std::string QHYCamera::CameraInfo::toString()
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
    toStr << "Camera is color: " << (isColor ? "Yes" : "No") << ", Bayer Pattern: " << bayerFormatToString() << std::endl;
    return toStr.str();
}

QHYCamera::~QHYCamera()
{
    close();
    release();
}

const uint8_t* QHYCamera::getFrame()
{
    std::chrono::high_resolution_clock timer;
    using fsec = std::chrono::duration<float>;
    uint32_t w, h, bpp, channels;
    static int numFrames = 0;
    static fsec fpsTime = fsec::zero();

    int tries = 0;
    auto start = timer.now();
    while (GetQHYCCDLiveFrame(pCamHandle, &w, &h, &bpp, &channels, pImgData) != QHYCCD_SUCCESS)
    {
        usleep(10000);
        if (++tries % 200 == 0)
            std::cout << "retries: " << tries << std::endl;
        // {
        //     fprintf(stderr, "Could not capture image\n");
        //     return nullptr;
        // }
    }
    auto stop = timer.now();
    fsec duration = (stop - start);
    ++numFrames;
    fpsTime += duration;

    //fprintf(stderr, "bpp: %d, channels: %d\n", bpp, channels);
    if (fpsTime.count() > 2.0f)
    {
        fprintf(stderr, "Capture duration: %.3f seconds, fps: %.3f\n", duration.count(), 1 / duration.count());
        numFrames = 0;
        fpsTime = fsec::zero();
    }
    return pImgData;
}

bool QHYCamera::getCameraInfo(std::string camId, CameraInfo &ci)
{
    // open camera
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

    // get overscan area
    int rc = GetQHYCCDOverScanArea(camHandle, &ci.overscanStartX, &ci.overscanStartY, &ci.overscanWidth, &ci.overscanHeight);
    if (rc != QHYCCD_SUCCESS)
    {
        std::cerr << "GetQHYCCDOverScanArea failure, camera id: " << camId << ", error: " << rc << std::endl;
        return false;
    }

    rc = GetQHYCCDOverScanArea(camHandle, &ci.effectiveStartX, &ci.effectiveStartY, &ci.effectiveWidth, &ci.effectiveHeight);
    if (rc != QHYCCD_SUCCESS)
    {
        std::cerr << "GetQHYCCDOverScanArea failure, camera id: " << camId << ", error: " << rc << std::endl;
        return false;
    }

    // get chip info
    rc = GetQHYCCDChipInfo(camHandle, &ci.chipWidthMM, &ci.chipHeightMM, &ci.maxImageWidth, &ci.maxImageHeight,
                            &ci.pixelWidthUM, &ci.pixelHeightUM, &ci.bpp);
    if (rc != QHYCCD_SUCCESS)
    {
        fprintf(stderr, "GetQHYCCDChipInfo failure, error: %d\n", rc);
        return false;
    }

    // check color camera
    ci.bayerFormat = IsQHYCCDControlAvailable(camHandle, CAM_COLOR);
    ci.isColor = (ci.bayerFormat == BAYER_GB || ci.bayerFormat == BAYER_GR || ci.bayerFormat == BAYER_BG || ci.bayerFormat == BAYER_RG);

    rc = CloseQHYCCD(camHandle);
    if (rc != QHYCCD_SUCCESS)
    {
        std::cerr << "Close QHYCCD failure, error: " << rc << std::endl;
    }
    std::cout << ci.toString() << std::endl;

    return true;
}

bool QHYCamera::scanCameras()
{
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
        const int rc = GetQHYCCDId(i, camId);
        if (rc == QHYCCD_SUCCESS)
        {
            CameraInfo ci;
            if (getCameraInfo(camId, ci))
            {
                m_cameras.push_back(ci);
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

bool QHYCamera::setControl(ControlParam controlParam, double value)
{
    int rc = IsQHYCCDControlAvailable(pCamHandle, (CONTROL_ID)controlParam);
    if (rc == QHYCCD_SUCCESS)
    {
        rc = SetQHYCCDParam(pCamHandle, (CONTROL_ID)controlParam, value);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "setControl failed: " << controlParam << std::endl;
            return false;
        }
    }
    return true;
}

bool QHYCamera::setBinMode(uint32_t binX, uint32_t binY)
{
    int rc = SetQHYCCDBinMode(pCamHandle, binX, binY);
    if (rc != QHYCCD_SUCCESS)
    {
        std::cerr << "SetQHYCCDBinMode failure, error: " << rc << std::endl;
        return false;
    }
    return true;
}

bool QHYCamera::setResolution(uint32_t startX, uint32_t startY, uint32_t width, uint32_t height)
{
    int rc = SetQHYCCDResolution(pCamHandle, startX, startY, width, height);
    if (QHYCCD_SUCCESS != rc)
    {
        std::cerr << "SetQHYCCDResolution failure, error: " << rc << std::endl;
        return false;
    }
    return true;
}

bool QHYCamera::init()
{
    EnableQHYCCDMessage(false);
    EnableQHYCCDLogFile(false);

    // init SDK
    int rc = InitQHYCCDResource();
    if (QHYCCD_SUCCESS != rc)
    {
        std::cerr << "Cannot initialize SDK resources, error: " << rc << std::endl;
        return false;
    }
    m_camInit = true;

    if (!scanCameras())
        return false;

    // HACK TO continue working
    // TODO move this away
    char *camId = (char *)m_cameras[0].id.c_str();

    // open camera
    pCamHandle = OpenQHYCCD(camId);
    if (pCamHandle == nullptr)
    {
        fprintf(stderr, "Open QHYCCD failure.\n");
        return false;
    }

    // check color camera
    if (m_cameras[0].isColor)
    {
        SetQHYCCDDebayerOnOff(pCamHandle, true);
        SetQHYCCDParam(pCamHandle, CONTROL_WBR, 76);
        SetQHYCCDParam(pCamHandle, CONTROL_WBG, 58);
        SetQHYCCDParam(pCamHandle, CONTROL_WBB, 64);
    }

    // set exposure time
    int EXPOSURE_TIME = 2000;
    if (!setControl(Exposure, EXPOSURE_TIME))
    {
        return false;
    }

    // N.B. SetQHYCCDStreamMode must be called immediately after CONTROL_EXPOSURE is SET
    // 1. Exposure
    // 2. Stream Mode
    // 3. Speed
    // 4. Traffic
    // 5. 8-bit

    rc = SetQHYCCDStreamMode(pCamHandle, 1);
    if (rc != QHYCCD_SUCCESS)
    {
        fprintf(stderr, "SetQHYCCDStreamMode failed: %d", rc);
        return false;
    }

    // initialize camera after setting stream mode
    rc = InitQHYCCD(pCamHandle);
    if (QHYCCD_SUCCESS != rc)
    {
        fprintf(stderr, "InitQHYCCD faililure, error: %d\n", rc);
        return false;
    }

    // check traffic
    int USB_TRAFFIC = 0;
    if (!setControl(UsbTraffic, USB_TRAFFIC))
    {
        return false;
    }

    // check speed
    int USB_SPEED = 1;
    if (!setControl(UsbSpeed, USB_SPEED))
    {
        return false;
    }

    // check gain
    int CHIP_GAIN = 30;
    if (!setControl(Gain, CHIP_GAIN))
    {
        return false;
    }

    // check offset
    int CHIP_OFFSET = 0;
    if (!setControl(Offset, CHIP_OFFSET))
    {
        return false;
    }

    // set image resolution
    unsigned int roiStartX = 0;
    unsigned int roiStartY = 0;
    unsigned int roiWidth = m_cameras[0].maxImageWidth;
    unsigned int roiHeight = m_cameras[0].maxImageHeight;
    if (!setResolution(roiStartX, roiStartY, roiWidth, roiHeight))
    {
        return false;
    }

    if (!setControl(TransferBits, 8))
    {
        return false;
    }

    // set binning mode
    int camBinX = 1;
    int camBinY = 1;
    if (!setBinMode(camBinX, camBinY))
    {
        return false;
    }

    // get requested memory lenght
    uint32_t length = GetQHYCCDMemLength(pCamHandle);
    if (length == 0)
    {
        fprintf(stderr, "Cannot allocate memory for frame.\n");
        return false;
    }

    pImgData = new uint8_t[length];
    memset(pImgData, 0, length);

    return true;
}

void QHYCamera::release()
{
    delete[] pImgData;
    pImgData = nullptr;

    // release sdk resources
    int rc = ReleaseQHYCCDResource();
    if (QHYCCD_SUCCESS != rc)
    {
        std::cerr << "Cannot release SDK resources, error: " << rc << std::endl;
    }
}

void QHYCamera::open()
{
    if ((pCamHandle == nullptr) && !init())
    {
        std::cerr << "Could not initialize Camera" << std::endl;
    }
    if (!m_camOpen)
    {
        int rc = BeginQHYCCDLive(pCamHandle);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "BeginQHYCCDLive failed: " << rc << std::endl;
            m_camOpen = false;
        }
        m_camOpen = true;
    }
}

void QHYCamera::close()
{
    if (m_camOpen)
    {
        StopQHYCCDLive(pCamHandle);
        SetQHYCCDStreamMode(pCamHandle, 0x0);

        // close camera handle
        int rc = CloseQHYCCD(pCamHandle);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "Close QHYCCD failure, error: " << rc << std::endl;
        }
        pCamHandle = nullptr;
        m_camOpen = false;
    }
}
