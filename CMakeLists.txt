cmake_minimum_required(VERSION 3.0.0)
project(boxdetection VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
include(CTest)
enable_testing()

find_package(OpenCV REQUIRED PATHS C:/opencv/build/x64/vc15/lib C:/opencv/build/x64/vc14/lib)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(boxdetection main.cpp)

target_link_libraries(boxdetection ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
