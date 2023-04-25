#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <chrono>

namespace sky360lib::utils
{
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double, std::nano>;

    struct ProfilerData
    {
        std::string name;
        TimePoint start_time;
        Duration duration;
        int count;

        double fps() const
        {
            return duration.count() > 0 ? (double)count / durationInSeconds() : 0.0;
        }

        double avgTimeInNs() const
        {
            return count > 0 ? duration.count() / count : 0.0;
        }

        double avgTimeInS() const
        {
            return count > 0 ? (duration.count() * 1e-9) / count : 0.0;
        }

        double durationInSeconds() const
        {
            return duration.count() * 1e-9;
        }
    };

    using DataMap = std::unordered_map<std::string, ProfilerData>;

    class Profiler
    {
    public:
        inline void start(const std::string &region)
        {
            auto time = Clock::now();
            if (m_profilerData.empty())
                start_time = time;

            auto& data = m_profilerData[region];
            if (data.name.empty())
                data.name = region;
            data.start_time = time;
        }

        inline void stop(const std::string &region)
        {
            auto& data = m_profilerData[region];
            Duration elapsed_time = Clock::now() - data.start_time;
            data.duration += elapsed_time;
            data.count++;
        }

        inline void reset()
        {
            m_profilerData.clear();
        }

        inline ProfilerData const& getData(const std::string &region) const
        {
            return m_profilerData.at(region);
        }

        inline DataMap const & getData() const
        {
            return m_profilerData;
        }

        void report() const
        {
            auto stop_time = Clock::now();
            for (const auto &entry : m_profilerData)
            {
                reportIndividual(entry.second, stop_time);
            }
        }

        void reportIndividual(const ProfilerData& data, TimePoint stop_time = Clock::now()) const
        {
            auto totalDuration = stop_time - start_time;
            std::cout << "Region: " << data.name
                        << ", Average Time (ns): " << data.avgTimeInNs()
                        << ", Average Time (s): " << data.avgTimeInS()
                        << ", Count: " << data.count
                        << ", FPS: " << data.fps()
                        << ", %: " << (data.duration / totalDuration)
                        << std::endl;
        }

    private:
        DataMap m_profilerData;
        TimePoint start_time;
    };
}