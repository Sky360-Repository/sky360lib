#pragma once

#include <opencv2/highgui.hpp>
#include <string>

namespace sky360lib::utils
{
    class TextWriter
    {
    public:
        TextWriter(cv::Scalar _color = cv::Scalar{0, 180, 180, 0}, int _numMaxLines = 32)
        : m_fontFace{cv::FONT_HERSHEY_COMPLEX}
        , m_numLines{_numMaxLines}
        , m_color{_color}
        , m_color16{_color[0] * 255, _color[1] * 255, _color[2] * 255, _color[3] * 255}
        {
        }

        void writeText(const cv::Mat _frame, std::string _text, int _line, bool _alignRight = false)
        {
            static const int maxHeight = getMaxTextHeight();
            const int fontScale = calcFontScale(maxHeight, _frame.size().height);
            const int thickness = (int)(3.5 * fontScale);
            const int height = calcHeight(_line, _frame.size().height);
            int posX = !_alignRight ? maxHeight : _frame.size().width - (getTextSize(_text, fontScale, thickness).width + maxHeight);

            cv::putText(_frame, _text, cv::Point(posX, height), m_fontFace, fontScale, _frame.elemSize1() == 1 ? m_color : m_color16, thickness, cv::LINE_AA);
        }

    private:
        const int m_fontFace;
        const int m_numLines;
        const cv::Scalar m_color;
        const cv::Scalar m_color16;

        inline cv::Size getTextSize(const std::string& _text, int _fontScale, int _thickness)
        {
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(_text, m_fontFace, _fontScale, _thickness, &baseline);
            textSize.height += baseline;
            return textSize;
        }

        inline int getMaxTextHeight()
        {
            const std::string text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(){}[]!|$#^0123456789";
            return getTextSize(text, 1.0, 5).height;
        }

        inline int calcFontScale(int _fontHeight, uint32_t _screenHeight)
        {
            const int lineHeight = _screenHeight / m_numLines;
            return lineHeight / _fontHeight;
        }

        inline int calcHeight(int _line, uint32_t _screenHeight)
        {
            const int lineHeight = _screenHeight / m_numLines;
            return _line * lineHeight;
        }
    };
}