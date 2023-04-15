include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{OpenCL_INCLUDE_DIRS})
include_directories($ENV{QHYCCD_SDK_INCLUDE_DIRS})

message(STATUS "QHY_SDK = $ENV{QHYCCD_SDK_INCLUDE_DIRS}")
#include_directories($ENV{CUDA_PATH}/include)
#message(STATUS "CUDA_PATH = $ENV{CUDA_PATH}")

if (USE_HALIDE)
    include_directories($ENV{Halide_DIR}/include)
    find_package(Halide REQUIRED)

    # Generator
    add_halide_generator(wmv_mono.generator
                        SOURCES "bgs/WeightedMovingVariance/WMVMonoGenerator.cpp"
                        LINK_LIBRARIES Halide::Tools)
    add_halide_generator(wmv_mono_threshold.generator
                        SOURCES "bgs/WeightedMovingVariance/WMVMonoThresholdGenerator.cpp"
                        LINK_LIBRARIES Halide::Tools)
    add_halide_generator(wmv_color.generator
                        SOURCES "bgs/WeightedMovingVariance/WMVColorGenerator.cpp"
                        LINK_LIBRARIES Halide::Tools)
    add_halide_generator(wmv_color_threshold.generator
                        SOURCES "bgs/WeightedMovingVariance/WMVColorThresholdGenerator.cpp"
                        LINK_LIBRARIES Halide::Tools)
    # add_halide_generator(vibe_mono.generator
    #                      SOURCES "bgs/vibe/VibeMonoGenerator.cpp"
    #                      LINK_LIBRARIES Halide::Tools)
    # Filters
    add_halide_library(wmv_mono FROM wmv_mono.generator
                    STMT wmv_mono_STMT
                    SCHEDULE wmv_mono_SCHEDULE)
    add_halide_library(wmv_mono_threshold FROM wmv_mono_threshold.generator
                    STMT wmv_mono_threshold_STMT
                    SCHEDULE wmv_mono_threshold_SCHEDULE)
    add_halide_library(wmv_color FROM wmv_color.generator
                    STMT wmv_color_STMT
                    SCHEDULE wmv_color_SCHEDULE)
    add_halide_library(wmv_color_threshold FROM wmv_color_threshold.generator
                    STMT wmv_color_threshold_STMT
                    SCHEDULE wmv_color_threshold_SCHEDULE)

    add_halide_library(wmv_mono_auto_schedule FROM wmv_mono.generator
                    GENERATOR wmv_mono
                    STMT wmv_mono_auto_schedule_STMT
                    SCHEDULE wmv_mono_auto_schedule_SCHEDULE
                    AUTOSCHEDULER Halide::Li2018)
    add_halide_library(wmv_mono_threshold_auto_schedule FROM wmv_mono_threshold.generator
                    GENERATOR wmv_mono_threshold
                    STMT wmv_mono_auto_schedule_STMT
                    SCHEDULE wmv_mono_threshold_auto_schedule_SCHEDULE
                    AUTOSCHEDULER Halide::Li2018)
    add_halide_library(wmv_color_auto_schedule FROM wmv_color.generator
                    GENERATOR wmv_color
                    STMT wmv_color_auto_schedule_STMT
                    SCHEDULE wmv_color_auto_schedule_SCHEDULE
                    AUTOSCHEDULER Halide::Li2018)
    add_halide_library(wmv_color_threshold_auto_schedule FROM wmv_color_threshold.generator
                    GENERATOR wmv_color_threshold
                    STMT wmv_color_threshold_auto_schedule_STMT
                    SCHEDULE wmv_color_threshold_auto_schedule_SCHEDULE
                    AUTOSCHEDULER Halide::Li2018)

    # add_halide_library(vibe_mono_auto_schedule FROM vibe_mono.generator
    #                    GENERATOR vibe_mono
    #                    STMT vibe_mono_auto_schedule_STMT
    #                    SCHEDULE vibe_mono_auto_schedule_SCHEDULE
    #                    AUTOSCHEDULER Halide::Li2018)

    # Three different auto-schedulers (for my Ryzen 7, the best was Li2018)
    #    AUTOSCHEDULER Halide::Mullapudi2016)
    #    AUTOSCHEDULER Halide::Adams2019)
    #    AUTOSCHEDULER Halide::Li2018)
    target_sources(sky360lib_api PRIVATE "bgs/WeightedMovingVariance/WeightedMovingVarianceHalide.cpp")
    
endif ()

find_package(OpenCL REQUIRED)

#project(sky360lib CUDA CXX)
add_library(sky360lib_api STATIC)

# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 11)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()
# if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
#     set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
# endif()
# message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
# SET(CUDA_SEPARABLE_COMPILATION ON)
# find_package(CUDA QUIET REQUIRED)

target_sources(
    sky360lib_api
        PRIVATE
            "bgs/CoreBgs.cpp"
            "bgs/vibe/Vibe.cpp"
            "bgs/vibe/VibeUtils.hpp" 
            "bgs/WeightedMovingVariance/WeightedMovingVariance.cpp" 
            "bgs/WeightedMovingVariance/WeightedMovingVarianceCL.cpp" 
            "blobs/connectedBlobDetection.cpp"
        PUBLIC
            "include/profiling.hpp" 
            "bgs/bgs.hpp"
)

target_sources(
    sky360lib_api
        PRIVATE
            "tracking/trackerCSRT.cpp"
            "tracking/trackerCSRTScaleEstimation.cpp"
            "tracking/trackerCSRTSegmentation.cpp"
            "tracking/trackerCSRTUtils.cpp"
            "tracking/trackerModel.cpp"
            "tracking/featureColorName.cpp"
        PUBLIC
            "tracking/tracker.hpp"
)

target_sources(
    sky360lib_api
        PRIVATE
            "camera/qhyCamera.cpp"
        PUBLIC
            "camera/qhyCamera.hpp"
)

target_link_libraries(sky360lib_api
                    PRIVATE
                        "${OpenCV_LIBS}"
                        # Halide::Tools
                        # wmv_mono
                        # wmv_mono_auto_schedule
                        # wmv_mono_threshold
                        # wmv_mono_threshold_auto_schedule
                        # wmv_color
                        # wmv_color_auto_schedule
                        # wmv_color_threshold
                        # wmv_color_threshold_auto_schedule
                        easy_profiler
                        OpenCL::OpenCL
                        qhyccd
                        )

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(sky360lib_api
            PRIVATE
                TBB::tbb
    )
endif ()

target_include_directories(
    sky360lib_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include" 
            "${CMAKE_SOURCE_DIR}/api/bgs" 
            "${CMAKE_SOURCE_DIR}/api/tracking"
            "${CMAKE_SOURCE_DIR}/api/blobs" 
            "${CMAKE_SOURCE_DIR}/api/camera" 
            "${CMAKE_SOURCE_DIR}/api/utils" 
)
