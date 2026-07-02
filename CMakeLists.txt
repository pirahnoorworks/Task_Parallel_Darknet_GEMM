cmake_minimum_required(VERSION 3.16)

project(darknet_xitao_dgemm_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_NATIVE_XITAO "Use installed XiTAO runtime instead of bundled compatibility runtime" OFF)

find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(darknet_xitao_dgemm_demo
  src/darknet_gemm.cpp
  src/main.cpp
)

target_include_directories(darknet_xitao_dgemm_demo PRIVATE include)
target_link_libraries(darknet_xitao_dgemm_demo PRIVATE Threads::Threads)

if(OpenMP_CXX_FOUND)
  target_link_libraries(darknet_xitao_dgemm_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

if(USE_NATIVE_XITAO)
  target_compile_definitions(darknet_xitao_dgemm_demo PRIVATE USE_NATIVE_XITAO)
  target_link_libraries(darknet_xitao_dgemm_demo PRIVATE xitao)
endif()

if(MSVC)
  target_compile_options(darknet_xitao_dgemm_demo PRIVATE /W4)
else()
  target_compile_options(darknet_xitao_dgemm_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
