cmake_minimum_required(VERSION 2.8)
project(pd2)
find_package(OpenCV REQUIRED)
add_executable(pd2 main.cpp imageClass.hpp imageClass.cpp)
target_link_libraries(pd2 ${OpenCV_LIBS})