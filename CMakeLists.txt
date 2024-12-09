cmake_minimum_required(VERSION 3.10)
project(MyProject)

find_package(OpenCV REQUIRED)

add_executable(test main.cpp)

target_link_libraries(test PRIVATE ${OpenCV_LIBS})
target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS})
