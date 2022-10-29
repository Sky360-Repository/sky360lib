include_directories(${OpenCV_INCLUDE_DIRS})

add_library(sky360lib_api STATIC)

target_sources(
    sky360lib_api
        PRIVATE
            "src/VibeBGS.cpp" "include/VibeBGSUtils.hpp" "src/WeightedMovingVariance.cpp"
        PUBLIC
            "include/VibeBGS.hpp" "include/WeightedMovingVariance.hpp"
)

target_include_directories(
    sky360lib_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include"
)

