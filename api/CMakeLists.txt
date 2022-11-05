include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{CUDA_PATH}/include)
message(STATUS "CUDA_PATH = $ENV{CUDA_PATH}/include")

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
            "bgs/WeightedMovingVariance/WMVCuda.cu"
        PUBLIC
            "include/profiling.hpp" "bgs/bgs.hpp"
)

target_include_directories(
    sky360lib_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include" "${CMAKE_SOURCE_DIR}/api/bgs"
)

