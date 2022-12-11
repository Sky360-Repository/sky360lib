include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{CUDA_PATH}/include)
include_directories($ENV{Halide_DIR}/include)
message(STATUS "CUDA_PATH = $ENV{CUDA_PATH}")

find_package(Halide REQUIRED)

# Generator
add_halide_generator(wmv_halide.generator
                     SOURCES "bgs/WeightedMovingVariance/WMVHalideGenerator.cpp"
                     LINK_LIBRARIES Halide::Tools)

# Filters
add_halide_library(wmv_halide FROM wmv_halide.generator
                   STMT wmv_halide_STMT
                   SCHEDULE wmv_halide_SCHEDULE)

add_halide_library(wmv_halide_auto_schedule FROM wmv_halide.generator
                   GENERATOR wmv_halide
                   STMT wmv_halide_auto_schedule_STMT
                   SCHEDULE wmv_halide_auto_schedule_SCHEDULE
                   AUTOSCHEDULER Halide::Li2018)
                # Three different auto-schedulers (for my Ryzen 7, the best was Li2018)
                #    AUTOSCHEDULER Halide::Mullapudi2016)
                #    AUTOSCHEDULER Halide::Adams2019)
                #    AUTOSCHEDULER Halide::Li2018)

add_library(sky360lib_api STATIC)

target_sources(
    sky360lib_api
        PRIVATE
            "include/pcg32.hpp" "include/core.hpp"
            "bgs/CoreBgs.hpp" "bgs/CoreBgs.cpp"
            "bgs/vibe/Vibe.cpp" "bgs/vibe/Vibe.hpp"
            "bgs/vibe/VibeUtils.hpp" 
            "bgs/WeightedMovingVariance/WeightedMovingVarianceUtils.hpp"
            "bgs/WeightedMovingVariance/WeightedMovingVariance.cpp" 
            "bgs/WeightedMovingVariance/WeightedMovingVariance.hpp"
            "bgs/WeightedMovingVariance/WeightedMovingVarianceCuda.cpp" 
            "bgs/WeightedMovingVariance/WeightedMovingVarianceCuda.hpp"
            "bgs/WeightedMovingVariance/WeightedMovingVarianceHalide.cpp"
            "bgs/WeightedMovingVariance/WeightedMovingVarianceHalide.hpp"
            "bgs/WeightedMovingVariance/WMVCuda.cu"
        PUBLIC
            "include/profiling.hpp" "bgs/bgs.hpp"
)

target_link_libraries(sky360lib_api
                      PRIVATE
                      Halide::Tools
                      wmv_halide
                      wmv_halide_auto_schedule)

target_include_directories(
    sky360lib_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include" "${CMAKE_SOURCE_DIR}/api/bgs"
)

