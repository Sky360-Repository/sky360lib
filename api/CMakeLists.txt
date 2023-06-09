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
            "camera/qhy_camera.cpp"
        PUBLIC
            "camera/qhy_camera.hpp"
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

 install(TARGETS sky360lib_api LIBRARY DESTINATION lib)
 set(HEADER_FILES 
    "${CMAKE_SOURCE_DIR}/api/include/coreUtils.hpp"
    "${CMAKE_SOURCE_DIR}/api/include/pcg32.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/bgs.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/CoreBgs.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/CoreParameters.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/vibe/Vibe.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/vibe/VibeUtils.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/WeightedMovingVariance/WeightedMovingVariance.hpp"
    "${CMAKE_SOURCE_DIR}/api/bgs/WeightedMovingVariance/WeightedMovingVarianceUtils.hpp"
    "${CMAKE_SOURCE_DIR}/api/blobs/connectedBlobDetection.hpp"
    "${CMAKE_SOURCE_DIR}/api/camera/qhy_camera.hpp"
    "${CMAKE_SOURCE_DIR}/api/camera/qhyccd_wrap.h"
    "${CMAKE_SOURCE_DIR}/api/utils/autoExposureControl.hpp"
    "${CMAKE_SOURCE_DIR}/api/utils/autoWhiteBalance.hpp"
    "${CMAKE_SOURCE_DIR}/api/utils/profiler.hpp"
    "${CMAKE_SOURCE_DIR}/api/utils/ringbuf.h"
    "${CMAKE_SOURCE_DIR}/api/utils/roi_mask_calculator.hpp"
    "${CMAKE_SOURCE_DIR}/api/utils/textWriter.hpp"
    "${CMAKE_SOURCE_DIR}/api/utils/utils.hpp"
    )

foreach(HEADER ${HEADER_FILES})
    file(RELATIVE_PATH RELATIVE_PATH_TO_HEADER ${CMAKE_SOURCE_DIR} ${HEADER})
    get_filename_component(DIR ${RELATIVE_PATH_TO_HEADER} DIRECTORY)

    install(DIRECTORY DESTINATION include/sky360lib/${DIR})

    install(FILES ${HEADER} DESTINATION ${DIR})
endforeach()
