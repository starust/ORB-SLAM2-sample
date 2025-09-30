cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM2_Demo)

set(CMAKE_CXX_STANDARD 14)

# vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "F:/openCV/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
