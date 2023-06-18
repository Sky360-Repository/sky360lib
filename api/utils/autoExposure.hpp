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
            m_max_exposure(150000),
            m_max_gain(25),
            m_min_gain(0),
            m_pid_controller(kp, ki, kd, [this] { return m_current_msv; }, [](double){})
        {
            m_pid_controller.setTarget(m_target_msv);
        }

        double get_target_msv() const { return m_target_msv; }
        double get_current_msv() const { return m_current_msv; }

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

            if (std::abs(error) > 0.03)
            {
                if (exposure + pidOutput <= m_max_exposure)
                {
                    exposure += pidOutput;
                }
                else if (gain + 1 <= m_max_gain)
                {
                    gain += 1;
                }
            }
            else
            {
                if (gain - 1 >= m_min_gain)
                {
                    gain -= 1;
                }
                else if (exposure + pidOutput >= m_min_exposure)
                {
                    exposure += pidOutput;
                }                
            }
        }

    private:
        double m_target_msv;
        double m_max_exposure;
        double m_min_exposure;
        double m_max_gain;
        double m_min_gain;
        double m_current_msv;
        PIDController<double> m_pid_controller;
        cv::RNG m_random;
    };
    
}