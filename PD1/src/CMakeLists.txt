cmake_minimum_required(VERSION 2.8)
project(pd1)
find_package(OpenCV REQUIRED)
add_executable(pd1 main.cpp imageClass.hpp imageClass.cpp functions.hpp functions.cpp)
target_link_libraries(pd1 ${OpenCV_LIBS})