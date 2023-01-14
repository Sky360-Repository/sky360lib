#pragma once

#include <string.h>
#include <sstream>
#include <iostream>

#include <qhyccd.h>

namespace sky360lib::camera
{
    class QHYCamera
    {
    public:
        struct CameraInfo
        {
            std::string id;
            std::string model;
            std::string serialNum;

            uint32_t overscanStartX;
            uint32_t overscanStartY;
            uint32_t overscanWidth;
            uint32_t overscanHeight;

            uint32_t effectiveStartX;
            uint32_t effectiveStartY;
            uint32_t effectiveWidth;
            uint32_t effectiveHeight;

            double chipWidthMM;
            double chipHeightMM;

            double pixelWidthUM;
            double pixelHeightUM;

            uint32_t maxImageWidth;
            uint32_t maxImageHeight;

            unsigned int bpp;

            uint32_t bayerFormat;
            bool isColor;

            std::string bayerFormatToString();

            std::string toString();
        };

        struct CameraParams
        {
            uint32_t roiStartX;
            uint32_t roiStartY;
            uint32_t roiSizeX;
            uint32_t roiSizeY;

            bool applyDebayer;
            double weightBalanceR;
            double weightBalanceG;
            double weightBalanceB;

            uint32_t exposureTime;

            enum StreamModeType
            {
                SingleFrame = 0,
                LiveFrame = 1
            };
            StreamModeType streamMode; 

            uint32_t usbTraffic;
            uint32_t usbSpeed;
            uint32_t gain;
            uint32_t offset;
            uint32_t binX;
            uint32_t binY;

            uint32_t transferBits;
        };

        enum ControlParam
        {
            Brightness = CONTROL_BRIGHTNESS, //!< image brightness
            Contrast = CONTROL_CONTRAST, //!< image contrast
            Exposure = CONTROL_EXPOSURE, //!< expose time (us)
            UsbTraffic = CONTROL_USBTRAFFIC, //!< hblank
            UsbSpeed = CONTROL_SPEED, //!< transfer speed
            Gain = CONTROL_GAIN, //!< camera gain
            Offset = CONTROL_OFFSET, //!< camera offset
            TransferBits = CONTROL_TRANSFERBIT, //!< image depth bits
            RedWB = CONTROL_WBR, //!< red of white balance
            BlueWB = CONTROL_WBB, //!< blue of white balance
            GreenWB = CONTROL_WBG, //!< the green of white balance
            Gamma = CONTROL_GAMMA, //!< screen gamma
            Channels = CONTROL_CHANNELS //!< image channels
        };

        QHYCamera();
        ~QHYCamera();

        const std::vector<CameraInfo>& getCameras();

        const uint8_t* getFrame();

        bool init();
        void release();

        bool open(std::string cameraId);
        void close();

        bool setControl(ControlParam controlParam, double value);
        bool debayer(bool enable);
        bool setBinMode(uint32_t binX, uint32_t binY);
        bool setResolution(uint32_t startX, uint32_t startY, uint32_t width, uint32_t height);

        uint32_t getMemoryNeededForFrame() const;

    private:
        qhyccd_handle *pCamHandle{nullptr};
        uint8_t *pImgData{nullptr};
        std::vector<CameraInfo> m_cameras;

        bool m_camInit{false};
        bool m_camOpen{false};

        bool getCameraInfo(std::string camId, CameraInfo &ci);
        bool scanCameras();
        bool allocBufferMemory();
        void releaseBufferMemory();
        bool setDefaultParams();
    };
}