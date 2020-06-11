
cmake_minimum_required(VERSION 2.8)
 
PROJECT(Circle)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(Circle MACOSX_BUNDLE Circle.cxx )
 
target_link_libraries(Circle ${VTK_LIBRARIES})
