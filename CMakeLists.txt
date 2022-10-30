cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/"
    "${CMAKE_SOURCE_DIR}/cmake/Modules/"
)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(
        FATAL_ERROR
        "In-source build not supported; generation should be done in a subdir, e.g.:  ${CMAKE_SOURCE_DIR}/build"
    )
endif ()

include(FrameworkUtils)
include(CheckFunctionExists)

project(sky360lib CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix (optional)")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
# if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Forced project build type" FORCE)
# endif (NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:DEBUG>")
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:_DEBUG>")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(USE_LINK_TIME_OPTIM "Enable link time optimization" ON)
option(USE_INLINE_INTRINSIC_FUNCS "Enable use of built-in inline intrinsic functions" ON)
option(USE_FAST_MATH "Enable fast math optimization" OFF)
option(USE_OPENMP "Enable OpenMP in internal implementations" ON)

find_package(OpenCV 4.0 REQUIRED)
message(STATUS "Found OpenCV >=4.0 at '${OpenCV_DIR}'")

if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_subdirectory(cmake/checks/simd)
# if (USE_NEON)
#     add_definitions(-mfpu=neon)
# endif ()
if (USE_FAST_MATH)
    add_definitions(-ffast-math)
endif ()
IF (NOT WIN32)
    add_definitions(-O3)
    add_definitions(-Ofast)
    add_definitions(-march=native)
    # add_definitions(-Wall)
    # add_definitions(-Wfatal-errors)
    # add_definitions(-Wextra)
    # add_definitions(-Wshadow)
    # add_definitions(-Werror)
    # add_definitions(-pedantic-errors)
    add_definitions(-Wno-missing-braces)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(TBB REQUIRED)  # for <execution> stdlib
    endif ()
ENDIF()
if (USE_LINK_TIME_OPTIM)
    IF (NOT WIN32)
        add_definitions(-flto)
    ENDIF()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        find_program(GCC_AR gcc-ar)
        if (GCC_AR)
            set(CMAKE_AR ${GCC_AR})
        endif ()
        find_program(GCC_RANLIB gcc-ranlib)
        if (GCC_RANLIB)
            set(CMAKE_RANLIB ${GCC_RANLIB})
        endif ()
    endif ()
endif ()

add_subdirectory(api)
add_subdirectory(apps)

file(GLOB_RECURSE bgs_python_src wrapper/python/*.cpp)
file(GLOB_RECURSE bgs_python_inc wrapper/python/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules/pybind11/include)
#include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})


#set(PYBIND11_LTO_CXX_FLAGS "")
#set(BGS_PYTHON_VERSION 3)
##set(PYBIND11_PYTHON_VERSION 2.7 3.5 3.6)
#set(PYBIND11_PYTHON_VERSION ${BGS_PYTHON_VERSION})
##find_package(pybind11 REQUIRED)
#add_subdirectory(modules/pybind11)

#pybind11_add_module(bgs_python ${bgs_python_src} ${bgs_python_inc})

#target_link_libraries(bgs_python PRIVATE bgslibrary_core ${OpenCV_LIBS} ${PYTHON_LIBRARY} pybind11::module)
#set_target_properties(bgs_python PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
#target_compile_definitions(bgs_python PRIVATE BGS_PYTHON_SUPPORT=1)

## Set the output library name to bgslibrary because that's what setup.py and distutils expects.
#set_property(TARGET bgs_python PROPERTY OUTPUT_NAME "pysky360lib")
