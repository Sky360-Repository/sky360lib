#include "qhyCamera.hpp"

#include <chrono>
#include <thread>

using namespace sky360lib::camera;

std::string QHYCamera::CameraInfo::bayerFormatToString()
{
    switch (bayerFormat)
    {
    case BAYER_GB: return "BAYER_GB";
    case BAYER_GR: return "BAYER_GR";
    case BAYER_BG: return "BAYER_BG";
    case BAYER_RG: return "BAYER_RG";
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
    toStr << "Bits per Pixel: " << bpp << std::endl;
    toStr << "Camera is color: " << (isColor ? "Yes" : "No") << ", Bayer Pattern: " << bayerFormatToString() << std::endl;
    toStr << "Bin modes: 1x1: " << (hasBin1x1Mode ? "Yes" : "No") << ", 2x2: " << (hasBin2x2Mode ? "Yes" : "No") << ", 3x3: " << (hasBin3x3Mode ? "Yes" : "No") << ", 4x4: " << (hasBin4x4Mode ? "Yes" : "No") << std::endl;
    return toStr.str();
}

QHYCamera::QHYCamera()
{
    EnableQHYCCDMessage(false);
    EnableQHYCCDLogFile(false);
}

QHYCamera::~QHYCamera()
{
    release();
}

const uint8_t *QHYCamera::getFrame()
{
    using fsec = std::chrono::duration<float>;
    std::chrono::high_resolution_clock timer;
    uint32_t w, h, bpp, channels;
    static fsec fpsTime = fsec::zero();

    allocBufferMemory();

    int tries = 0;
    auto start = timer.now();
    while (GetQHYCCDSingleFrame(pCamHandle, &w, &h, &bpp, &channels, m_pImgData) != QHYCCD_SUCCESS)
    {
        usleep(10000);
        if (++tries > 100)
        {
            std::cout << "retries: " << tries << ", aborting." << std::endl;
            return nullptr;
        }
    }
    auto stop = timer.now();
    fsec duration = (stop - start);
    m_lastFrameCaptureTime = duration.count();

    return m_pImgData;
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
        fprintf(stderr, "GetQHYCCDChipInfo failure, error: %d\n", rc);
        return false;
    }

    ci.bayerFormat = IsQHYCCDControlAvailable(camHandle, CAM_COLOR);
    ci.isColor = (ci.bayerFormat == BAYER_GB || ci.bayerFormat == BAYER_GR || ci.bayerFormat == BAYER_BG || ci.bayerFormat == BAYER_RG);

    ci.hasBin1x1Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN1X1MODE) == QHYCCD_SUCCESS;
    ci.hasBin2x2Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN2X2MODE) == QHYCCD_SUCCESS;
    ci.hasBin3x3Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN3X3MODE) == QHYCCD_SUCCESS;
    ci.hasBin4x4Mode = IsQHYCCDControlAvailable(camHandle, CAM_BIN4X4MODE) == QHYCCD_SUCCESS;

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

const std::vector<QHYCamera::CameraInfo> &QHYCamera::getCameras()
{
    scanCameras();
    return m_cameras;
}

bool QHYCamera::setControl(ControlParam controlParam, double value)
{
    uint32_t rc = IsQHYCCDControlAvailable(pCamHandle, (CONTROL_ID)controlParam);
    if (rc == QHYCCD_SUCCESS)
    {
        rc = SetQHYCCDParam(pCamHandle, (CONTROL_ID)controlParam, value);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "setControl failed: " << controlParam << std::endl;
            return false;
        }
        releaseBufferMemory();
    }
    else
    {
        std::cout << "Control not available to change: " << controlParam << std::endl;
    }
    return true;
}

bool QHYCamera::debayer(bool enable)
{
    if (SetQHYCCDDebayerOnOff(pCamHandle, enable) != QHYCCD_SUCCESS)
    {
        std::cerr << "SetQHYCCDDebayerOnOff failure" << std::endl;
        return false;
    }
    releaseBufferMemory();

    return true;
}

bool QHYCamera::setBinMode(uint32_t binX, uint32_t binY)
{
    uint32_t rc = SetQHYCCDBinMode(pCamHandle, binX, binY);
    if (rc != QHYCCD_SUCCESS)
    {
        std::cerr << "SetQHYCCDBinMode failure, error: " << rc << std::endl;
        return false;
    }
    releaseBufferMemory();

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
    releaseBufferMemory();

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
        // init SDK
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

bool QHYCamera::setDefaultParams()
{
    // set exposure time
    int EXPOSURE_TIME = 2000;
    if (!setControl(Exposure, EXPOSURE_TIME))
    {
        return false;
    }

    // check traffic
    int USB_TRAFFIC = 1;
    if (!setControl(UsbTraffic, USB_TRAFFIC))
    {
        return false;
    }

    // check speed
    int USB_SPEED = 0;
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
    if (!setResolution(0, 0, m_cameras[0].maxImageWidth, m_cameras[0].maxImageHeight))
    {
        return false;
    }

    if (!setControl(TransferBits, 16))
    {
        return false;
    }

    // set binning mode
    if (!setBinMode(1, 1))
    {
        return false;
    }

    allocBufferMemory();

    return true;
}

void QHYCamera::release()
{
    if (m_camOpen)
    {
        close();
    }

    // uint32_t rc = CloseQHYCCD(pCamHandle);
    // if (rc != QHYCCD_SUCCESS)
    // {
    //     std::cerr << "Close QHYCCD failure, error: " << rc << std::endl;
    // }

    // releaseBufferMemory();

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
    // get requested memory lenght
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

bool QHYCamera::open(std::string cameraId)
{
    // open camera
    if (!m_camInit && !init())
    {
        return false;
    }
    if (!m_camOpen)
    {
        if (cameraId.empty())
        {
            if (!scanCameras())
                return false;
            cameraId = m_cameras[0].id;
        }

        pCamHandle = OpenQHYCCD((char*)cameraId.c_str());
        if (pCamHandle == nullptr)
        {
            std::cerr << "Open QHYCCD failure." << std::endl;
            return false;
        }

        uint32_t rc = InitQHYCCD(pCamHandle);
        if (QHYCCD_SUCCESS != rc)
        {
            fprintf(stderr, "InitQHYCCD faililure, error: %d\n", rc);
            return false;
        }

        setDefaultParams();

        rc = ExpQHYCCDSingleFrame(pCamHandle);
        if (rc != QHYCCD_ERROR)
        {
            if (rc == QHYCCD_READ_DIRECTLY)
            {
                usleep(200);
            }
            m_camOpen = true;
        }
        else
        {
            std::cerr << "ExpQHYCCDSingleFrame failed: " << rc << std::endl;
            m_camOpen = false;
        }
        // int rc = BeginQHYCCDLive(pCamHandle);
        //  if (rc != QHYCCD_SUCCESS)
        //  {
        //      std::cerr << "ExpQHYCCDSingleFrame failed: " << rc << std::endl;
        //      m_camOpen = false;
        //  }
    }
    return m_camOpen;
}

void QHYCamera::close()
{
    if (m_camOpen)
    {
        uint32_t rc = CancelQHYCCDExposingAndReadout(pCamHandle);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "CancelQHYCCDExposingAndReadout failure, error: " << rc << std::endl;
        }
        // StopQHYCCDLive(pCamHandle);
        // SetQHYCCDStreamMode(pCamHandle, 0x0);

        // close camera handle
        rc = CloseQHYCCD(pCamHandle);
        if (rc != QHYCCD_SUCCESS)
        {
            std::cerr << "Close QHYCCD failure, error: " << rc << std::endl;
        }
        releaseBufferMemory();

        pCamHandle = nullptr;
        m_camOpen = false;
    }
}
