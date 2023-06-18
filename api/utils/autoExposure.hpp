#pragma once

#include <opencv2/opencv.hpp>
#include "PID.h"
#include <functional> 
#include <limits>

namespace sky360lib::utils
{
    class AutoExposure 
    {
    public:
        AutoExposure(double targetBrightness, double kp, double ki, double kd)
            : m_target_msv(targetBrightness),
            m_max_exposure(50000),
            m_max_gain(25),
            m_min_gain(0),
            m_is_night(false),
            m_pid_controller(kp, ki, kd, [this] { return m_current_msv; }, [](double){})
        {
            m_pid_controller.setTarget(m_target_msv);
        }

        double get_target_msv() const { return m_target_msv; }
        double get_current_msv() const { return m_current_msv; }
        bool is_day() const { return !m_is_night; }

        void set_target_msv(double target_msv)
        {
            m_target_msv = target_msv;
            m_pid_controller.setTarget(m_target_msv);
        }

        void update(double msv, double& exposure, double& gain)
        {
            m_current_msv = msv;
            m_pid_controller.tick();
            double pidOutput = m_pid_controller.getOutput();
            double error = m_pid_controller.getError();
            double signOfPidOutput = std::signbit(pidOutput) ? -1.0 : 1.0;

            if (std::abs(error) > 0.04)
            {
                if (error > 0) // Light is decreasing
                {
                    if (exposure < m_max_exposure)
                    {
                        exposure = std::clamp(exposure += pidOutput, m_min_exposure, m_max_exposure);
                    }
                    else
                    {
                        gain = std::clamp(gain + signOfPidOutput, m_min_gain, m_max_gain);
                    }
                }
                else // Light is increasing
                {
                    if (gain > m_min_gain)
                    {
                        gain = std::clamp(gain + signOfPidOutput, m_min_gain, m_max_gain);
                    }
                    else
                    {
                        exposure = std::clamp(exposure += pidOutput, m_min_exposure, m_max_exposure);
                    }
                }
            }

            m_is_night = (exposure >= m_max_exposure) && (gain > m_min_gain);
        }

    private:
        double m_target_msv;
        double m_max_exposure;
        double m_min_exposure;
        double m_max_gain;
        double m_min_gain;
        double m_current_msv;
        bool m_is_night;
        PIDController<double> m_pid_controller;
        cv::RNG m_random;
    };
    
}