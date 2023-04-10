#pragma once

#include <string.h>
#include <sstream>
#include <map>
#include <iostream>

#include <qhyccd.h>
#include <opencv2/opencv.hpp>

namespace sky360lib::camera
{
    class QHYCamera
    {
    public:
    static const int DEFAULT_CAPTURE_RETRIES = 1000;
        enum BinMode
        {
            Bin_1x1 = 1,
            Bin_2x2 = 2,
            Bin_3x3 = 3,
            Bin_4x4 = 4
        };
        enum StreamModeType
        {
            SingleFrame = 0,
            LiveFrame = 1
        };

        struct CameraInfo
        {
            struct paramLimits
            {
                double min;
                double max;
                double step;
            };

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

            bool hasBin1x1Mode;
            bool hasBin2x2Mode;
            bool hasBin3x3Mode;
            bool hasBin4x4Mode;

            paramLimits gainLimits;
            paramLimits offsetLimits;
            paramLimits usbTrafficLimits;
            paramLimits redWBLimits;
            paramLimits greenWBLimits;
            paramLimits blueWBLimits;

            std::string bayerFormatToString() const;

            std::string toString() const;
        };

        struct CameraParams
        {
            uint32_t roiStartX;
            uint32_t roiStartY;
            uint32_t roiWidth;
            uint32_t roiHeight;

            bool applyDebayer;
            double redWB;
            double greenWB;
            double blueWB;

            uint32_t exposureTime;
            double contrast;
            double brightness;
            double gamma;

            StreamModeType streamMode; 

            uint32_t channels;
            uint32_t usbTraffic;
            uint32_t usbSpeed;
            uint32_t gain;
            uint32_t offset;
            BinMode binMode;

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

        void setDebugInfo(bool _enable);

        const std::map<std::string, CameraInfo>& getCameras();

        const uint8_t* getFrame();
        bool getFrame(cv::Mat& frame, bool debayer);
        cv::Mat getFrameRet(bool debayer);

        void debayerImage(const cv::Mat& imageIn, cv::Mat& imageOut) const;

        double getLastFrameCaptureTime() const;
        CameraInfo const * getCameraInfo() const;
        const CameraParams& getCameraParams() const;

        //uint32_t GetQHYCCDHumidity(qhyccd_handle *handle,double *hd);

        //uint32_t GetQHYCCDCameraStatus(qhyccd_handle *handle,uint8_t *buf); 
        // buf[0] buf[1] buf[2] buf[3]
        // 00 fe 81 74:idle,camera don't expose and readout
        // 01 fe 81 74:waiting,a span before starting expose,very short
        // 02 fe 81 74:exposing,open shutter to start expose,and will close shutter
        // after expose
        // 03 fe 81 74:read out image data

        //uint32_t ControlQHYCCDTemp(qhyccd_handle *handle,double targettemp);

        bool init();
        void release();

        bool open(std::string cameraId);
        void close();

        bool beginExposing();
        void endExposing();

        bool setControl(ControlParam controlParam, double value, bool force = false);
        bool setDebayer(bool enable);
        bool setBinMode(BinMode mode);
        bool setResolution(uint32_t startX, uint32_t startY, uint32_t width, uint32_t height);
        bool setStreamMode(StreamModeType mode);

        uint32_t getMemoryNeededForFrame() const;

    private:
        bool m_debugInfo;
        std::string m_camId;
        qhyccd_handle *pCamHandle{nullptr};
        uint8_t *m_pImgData{nullptr};
        std::map<std::string, CameraInfo> m_cameras;
        CameraParams m_params;
        CameraInfo* m_currentInfo;
        double m_lastFrameCaptureTime;

        bool m_camInit{false};
        bool m_camOpen{false};
        bool m_isExposing{false};
        bool m_defaultSet{false};

        bool fillCameraInfo(std::string camId, CameraInfo &ci);
        bool scanCameras();
        bool allocBufferMemory();
        void releaseBufferMemory();
        void setDefaultParams();
        void applyParams();
        bool getSingle(uint32_t *w, uint32_t *h, uint32_t *bpp, uint32_t *channels, uint8_t *imgData);
        bool getLive(uint32_t *w, uint32_t *h, uint32_t *bpp, uint32_t *channels, uint8_t *imgData);
    };
}