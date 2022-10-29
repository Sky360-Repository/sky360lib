#pragma once

#ifdef _WIN32

#include <windows.h>

double clockFrequency;

inline double initFrequency() {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    clockFrequency = 1.0 / frequency.QuadPart;
    return clockFrequency;
}

inline double getAbsoluteTime() {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * clockFrequency;
}

#else

#include <sys/time.h>

inline double initFrequency() {
    return 0;
}

inline double getAbsoluteTime() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 0.000000001;
}

#endif