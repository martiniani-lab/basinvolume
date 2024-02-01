
cmake_minimum_required(VERSION 3.5)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pele_tests)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CXX)

# TODO: Add a build type option specifically for memory leak testing
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# set OpenMP
find_package(OpenMP)
message("${OpenMP_C_FLAGS}")
message("${OpenMP_CXX_FLAGS}")

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# include LAPACK AND BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL Coverage)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG -fprofile-arcs -ftest-coverage")
endif()

message(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES})

# find conda path for python header
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
message("PYTHON INCLUDE DIR" ${PYTHON_INCLUDE_DIR})

set(EXT_INSTALL_PATH "../../../pele/extern/install")

set(LAPACKE_INCLUDE_DIR ${EXT_INSTALL_PATH}/include/Eigen/src/misc)

# set(SUNDIALS_EXTRA_LIBS  -lm /home/praharsh/anaconda3/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING "Additional libraries")
set(SUNDIALS_INCLUDE_DIR
    ${EXT_INSTALL_PATH}/include
    CACHE PATH "Location of SUNDIALS header files")
set(SUNDIALS_LIBRARY_DIR
    ${EXT_INSTALL_PATH}/lib
    CACHE PATH "Location of SUNDIALS libraries")

message("SUNDIALS_INCLUDE_DIR ${SUNDIALS_INCLUDE_DIR}")

# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
    sundials_cvode ${SUNDIALS_LIBRARY_DIR}
    DOC "CVODE library")

find_library(SUNDIALS_NVEC_LIB
    sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
    DOC "NVECTOR_SERIAL library")

find_library(SUNDIALS_MANYVEC_LIB
    sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR}
    DOC "NVECTOR_MANYVECTOR library")

set(SUNDIALS_LIBRARIES
    -L${SUNDIALS_LIBRARY_DIR}
    ${SUNDIALS_SOLVER_LIB}
    ${SUNDIALS_NVEC_LIB}
    ${SUNDIALS_MANYVEC_LIB}
    ${SUNDIALS_EXTRA_LIBS})

# cmake_policy(SET CMP0015 NEW)

# Add and compile the gtest library
set(gtest_src ../../../pele/cpp_tests/gtest)
set(gtest_binary_dir googletest)
set(gtest_include ${gtest_src}/include/)
include_directories(${gtest_include})
message("gtest include directory ${gtest_include}")
add_subdirectory(../../../pele/cpp_tests/gtest ${gtest_binary_dir})
set(gtest_libs gtest gtest_main)
message("gtest_libs ${gtest_libs}")

# For CVODE/Mixed Descent
include_directories(BEFORE SYSTEM ${LAPACKE_INCLUDE_DIR})
include_directories(${SUNDIALS_INCLUDE_DIR})

# set the basinvolume include directory
set(basinvolume_include ../../source)
include_directories(${basinvolume_include})
message("basinvolume include directory ${basinvolume_include}")

# set the PyCG_DESCENT include directory
set(pycgd_include ../../../PyCG_DESCENT/source)
include_directories(${pycgd_include})

# include_directories(${cgd_include})
message("pycgd_include include directory ${pycgd_include}")

# set the pele include directory
set(pele_include ../../../pele/source)
include_directories(${pele_include})
message("pele include directory ${pele_include}")

# set mcpele include directory
set(mcpele_include ../../../mcpele/source)
include_directories(${mcpele_include})
message("mcpele include directory ${mcpele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.c*)
add_library(pele_lib ${pele_sources})

# This path needs fixing as soon as the mc cpp files have been moved.
# build the mcpele library
FILE(GLOB mcpele_sources ${mcpele_include}/*.c*)
add_library(mcpele_lib ${mcpele_sources})

# build the PyCG_DESCENT library
FILE(GLOB pycgd_sources ${pycgd_include}/CG_DESCENT/*.c*)
add_library(pycgd_lib ${pycgd_sources})

# build the basinvolume library
FILE(GLOB basinvolume_sources ${basinvolume_include}/*.cpp)
add_library(basinvolume_lib ${basinvolume_sources})

# get all the source files
FILE(GLOB sources *.cpp)

add_executable(test_main ${sources})

# target_link_libraries(test_main basinvolume_lib gtest gtest_main pthread)
# target_link_libraries(test_main pele_lib mcpele_lib basinvolume_lib gtest gtest_main pthread)
# target_link_libraries(test_main pele_lib gtest gtest_main pthread)
target_link_libraries(test_main basinvolume_lib pycgd_lib mcpele_lib pele_lib gtest gtest_main pthread)
