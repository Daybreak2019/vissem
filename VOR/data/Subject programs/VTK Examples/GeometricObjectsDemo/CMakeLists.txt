
cmake_minimum_required(VERSION 2.8)
 
PROJECT(GeometricObjectsDemo)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(GeometricObjectsDemo MACOSX_BUNDLE GeometricObjectsDemo.cxx )
 
target_link_libraries(GeometricObjectsDemo ${VTK_LIBRARIES})
