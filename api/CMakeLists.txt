include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{QHYCCD_SDK_INCLUDE_DIRS})

# message(STATUS "QHY_SDK = $ENV{QHYCCD_SDK_INCLUDE_DIRS}")

add_library(sky360lib_api STATIC)

if(MSVC)
  target_compile_options(sky360lib_api PRIVATE /EHsc)
endif()

target_sources(
    sky360lib_api
        PRIVATE
            "bgs/CoreBgs.cpp"
            "bgs/vibe/Vibe.cpp"
            "bgs/vibe/VibeUtils.hpp" 
            "bgs/WeightedMovingVariance/WeightedMovingVariance.cpp" 
            "blobs/connectedBlobDetection.cpp"
        PUBLIC
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
            "camera/qhy_camera.cpp"
        PUBLIC
            "camera/qhyCamera.hpp"
)

target_link_libraries(sky360lib_api
                    PRIVATE
                        "${OpenCV_LIBS}"
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
