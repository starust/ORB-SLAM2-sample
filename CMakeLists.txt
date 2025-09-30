cmake_minimum_required(VERSION 3.15)
project(ORB_SLAM2_SAMPLE)

# 告诉 CMake vcpkg 工具链
set(CMAKE_TOOLCHAIN_FILE "D:/openCV/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# 告诉 CMake 依赖的安装位置
set(OpenCV_DIR "D:/openCV/vcpkg/installed/x64-windows/share/opencv4")
set(Protobuf_DIR "D:/openCV/vcpkg/installed/x64-windows/share/protobuf")
set(quirc_DIR "D:/openCV/vcpkg/installed/x64-windows/share/quirc")
set(absl_DIR "D:/openCV/vcpkg/installed/x64-windows/share/absl")
set(utf8_range_DIR "D:/openCV/vcpkg/installed/x64-windows/share/utf8_range")
set(TIFF_DIR "D:/openCV/vcpkg/installed/x64-windows/share/tiff")


# 查找 OpenCV
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(quirc REQUIRED)
find_package(absl REQUIRED)
find_package(utf8_range REQUIRED)

# 你的源文件
add_executable(main main.cpp)

# 链接 OpenCV 库
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
