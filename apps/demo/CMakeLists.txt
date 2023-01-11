include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{OpenCL_INCLUDE_DIRS})
# message(STATUS "OpenCL_INCLUDE_DIRS = $ENV{OpenCL_INCLUDE_DIRS}")

find_package(OpenCL REQUIRED)

# LIST(APPEND CMAKE_MODULE_PATH "~/sdk_Arm64_22.10.14/usr/local/cmake_modules/")
# find_package(QHY REQUIRED)

add_executable(
    sky360lib_demo
        "src/demo_main.cpp"
        "src/demoTracker.hpp"
)

target_include_directories(
    sky360lib_demo
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/api/include>"
)

target_link_libraries(
    sky360lib_demo
        PUBLIC
            "${OpenCV_LIBS}"
            sky360lib_api
            easy_profiler
            OpenCL::OpenCL
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(  # for <execution> stdlib
    sky360lib_demo
            PRIVATE
                TBB::tbb
    )
endif ()

set_target_properties(
    sky360lib_demo
        PROPERTIES
            FOLDER "apps"
)

install(
    TARGETS sky360lib_demo
    RUNTIME DESTINATION "bin"
    COMPONENT "apps"
)

set(VIDEOTEST ${CMAKE_SOURCE_DIR}/Dahua-20220901-184734.mp4)
if (EXISTS ${VIDEOTEST})
    message(STATUS "Copying Video Dahua-20220901-184734.mp4")
    configure_file(${VIDEOTEST} "${CMAKE_BINARY_DIR}/bin/Dahua-20220901-184734.mp4" COPYONLY)
endif ()
