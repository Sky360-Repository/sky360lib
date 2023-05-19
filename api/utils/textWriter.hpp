#pragma once

#include <opencv2/highgui.hpp>
#include <string>

namespace sky360lib::utils
{
    class TextWriter
    {
    public:
        TextWriter(cv::Scalar _color = cv::Scalar{80, 140, 190, 0}, int _numMaxLines = 32, double _thickness_scale = 3.5)
        : m_fontFace{cv::FONT_HERSHEY_COMPLEX}
        , m_numLines{_numMaxLines}
        , m_color{_color}
        , m_color16{_color[0] * 255, _color[1] * 255, _color[2] * 255, _color[3] * 255}
        , m_thickness_scale{_thickness_scale}
        {
            m_max_height = getMaxTextHeight();
            m_horizontal_padding = m_max_height / 2;
        }

        void writeText(const cv::Mat _frame, std::string _text, int _line, bool _alignRight = false) const
        {
            const double fontScale = calcFontScale(m_max_height, _frame.size().height);
            const int thickness = (int)(m_thickness_scale * fontScale);
            const int height = calcHeight(_line, _frame.size().height);
            int posX = !_alignRight ? m_horizontal_padding : _frame.size().width - (getTextSize(_text, fontScale, thickness).width + m_horizontal_padding);

            cv::putText(_frame, _text, cv::Point(posX, height), m_fontFace, fontScale, _frame.elemSize1() == 1 ? m_color : m_color16, thickness, cv::LINE_AA);
        }

    private:
        const int m_fontFace;
        const int m_numLines;
        const cv::Scalar m_color;
        const cv::Scalar m_color16;
        double m_thickness_scale;
        int m_max_height;
        int m_horizontal_padding;

        inline cv::Size getTextSize(const std::string& _text, double _fontScale, int _thickness) const
        {
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(_text, m_fontFace, _fontScale, _thickness, &baseline);
            textSize.height += baseline;
            return textSize;
        }

        inline int getMaxTextHeight() const
        {
            const std::string text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(){}[]!|$#^0123456789";
            return getTextSize(text, 1.0, 5).height;
        }

        inline double calcFontScale(int _fontHeight, uint32_t _screenHeight) const
        {
            const double lineHeight = (double)_screenHeight / (double)m_numLines;
            return lineHeight / (double)_fontHeight;
        }

        inline int calcHeight(int _line, uint32_t _screenHeight) const
        {
            const int lineHeight = _screenHeight / m_numLines;
            return _line * lineHeight;
        }
    };
}