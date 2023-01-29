include_directories(${OpenCV_INCLUDE_DIRS})

# sky360lib_video
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
)

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

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(  # for <execution> stdlib
    sky360lib_demo
            PRIVATE
                TBB::tbb
    )
endif ()

# sky360lib_demo_qhy_camera
add_executable(
    sky360lib_demo_qhy_camera
        "src/demo_qhy_camera.cpp"
)

target_include_directories(
    sky360lib_demo_qhy_camera
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/api/include>"
)

target_link_libraries(
    sky360lib_demo_qhy_camera
        PUBLIC
            "${OpenCV_LIBS}"
            sky360lib_api
            easy_profiler
)

set_target_properties(
    sky360lib_demo_qhy_camera
        PROPERTIES
            FOLDER "apps"
)

install(
    TARGETS sky360lib_demo_qhy_camera
    RUNTIME DESTINATION "bin"
    COMPONENT "apps"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(  # for <execution> stdlib
    sky360lib_demo_qhy_camera
            PRIVATE
                TBB::tbb
    )
endif ()

# sky360lib_qhy_camera_loop
add_executable(
    sky360lib_qhy_camera_loop
        "src/qhy_camera_loop.cpp"
)

target_include_directories(
    sky360lib_qhy_camera_loop
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/api/include>"
)

target_link_libraries(
    sky360lib_qhy_camera_loop
        PUBLIC
            "${OpenCV_LIBS}"
            sky360lib_api
            easy_profiler
)

set_target_properties(
    sky360lib_qhy_camera_loop
        PROPERTIES
            FOLDER "apps"
)

install(
    TARGETS sky360lib_qhy_camera_loop
    RUNTIME DESTINATION "bin"
    COMPONENT "apps"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(  # for <execution> stdlib
    sky360lib_qhy_camera_loop
            PRIVATE
                TBB::tbb
    )
endif ()

set(VIDEOTEST ${CMAKE_SOURCE_DIR}/Dahua-20220901-184734.mp4)
if (EXISTS ${VIDEOTEST})
    message(STATUS "Copying Video Dahua-20220901-184734.mp4")
    configure_file(${VIDEOTEST} "${CMAKE_BINARY_DIR}/bin/Dahua-20220901-184734.mp4" COPYONLY)
endif ()
