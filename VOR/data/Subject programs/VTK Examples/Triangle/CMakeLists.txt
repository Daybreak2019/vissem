
cmake_minimum_required(VERSION 2.8)
 
PROJECT(Triangle)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(Triangle MACOSX_BUNDLE Triangle.cxx )
 
target_link_libraries(Triangle ${VTK_LIBRARIES})
