include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
    sky360lib_demo
        "src/demo_main.cpp"
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
)

# target_link_libraries(sky360lib_demo PUBLIC cublas cudart_static cufilt cusparse nppidei nppist nvblas nvrtc
# cublasLt curand nppc nppif nppisu nvjpeg nvrtc_static
# cuda cufft cusolver nppial nppig nppitc nvml nvrtc-builtins_static
# cudadevrt cufftw cusolverMg nppicc nppim npps nvptxcompiler_static OpenCL)

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
