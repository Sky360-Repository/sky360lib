include_directories(${OpenCV_INCLUDE_DIRS})

add_library(sky360lib_api STATIC)

target_sources(
    sky360lib_api
        PRIVATE
            "bgs/vibe/Vibe.cpp" "bgs/vibe/VibeUtils.hpp" "bgs/WeightedMovingVariance/WeightedMovingVariance.cpp" "include/pcg32.hpp" "include/core.hpp" 
            "bgs/vibe/Vibe.hpp" "bgs/WeightedMovingVariance/WeightedMovingVariance.hpp" 
        PUBLIC
            "include/profiling.hpp" "bgs/bgs.hpp"
)

target_include_directories(
    sky360lib_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include" "${CMAKE_SOURCE_DIR}/api/bgs"
)

