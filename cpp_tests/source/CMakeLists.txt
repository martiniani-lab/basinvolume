cmake_minimum_required(VERSION 2.8)

enable_language(CXX)
ADD_DEFINITIONS(-std=c++0x)
ADD_DEFINITIONS(-Wall)

#cmake_policy(SET CMP0015 NEW)

# Add and compile the gtest library
set(gtest_src ../gtest)
set(gtest_binary_dir googletest)
set(gtest_include ${gtest_src}/include/)
include_directories(${gtest_include})
message("gtest include directory ${gtest_include}")
add_subdirectory(../gtest ${gtest_binary_dir})
set(gtest_libs gtest gtest_main)
message("gtest_libs ${gtest_libs}")

# set the basinvolume include directory
set(basinvolume_include ../../source)
include_directories(${basinvolume_include})
message("basinvolume include directory ${basinvolume_include}")

# set the pele include directory
set(pele_include ../../../pele/source)
include_directories(${pele_include})
message("pele include directory ${pele_include}")

## build the basinvolume library
#FILE(GLOB basinvolume_sources ${basinvolume_include}/*.cpp)
#add_library(basinvolume_lib ${basinvolume_sources})

# get all the source files
FILE(GLOB sources *.cpp)

add_executable(test_main ${sources})
#target_link_libraries(test_main basinvolume_lib gtest gtest_main pthread)
target_link_libraries(test_main gtest gtest_main pthread)
