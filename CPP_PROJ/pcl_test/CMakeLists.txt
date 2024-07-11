cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_project)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the PCL package with visualization component
find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set build type to Debug to include debug symbols
set(CMAKE_BUILD_TYPE Debug)

add_executable(pcd_write pcd_write.cpp)
target_link_libraries(pcd_write ${PCL_LIBRARIES})

add_executable(pcd_read pcd_read.cpp)
target_link_libraries(pcd_read ${PCL_LIBRARIES})

add_executable(bin_read bin_read.cpp)
target_link_libraries(bin_read ${PCL_LIBRARIES})