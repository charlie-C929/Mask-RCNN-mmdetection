cmake_minimum_required(VERSION 3.5)
project(inference_C++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{ONNXRUNTIME_DIR}/include)
include_directories(./include)

message("$ENV{ONNXRUNTIME_DIR}")
message("${OpenCV_INCLUDE_DIRS}")

link_directories($ENV{ONNXRUNTIME_DIR}/lib)

add_executable(runDet runDet.cpp)
target_link_libraries(runDet ${OpenCV_LIBS})
message("$ENV{ONNXRUNTIME_DIR}/lib")
target_link_libraries(runDet $ENV{ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
target_link_libraries(runDet $ENV{ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.8.1)