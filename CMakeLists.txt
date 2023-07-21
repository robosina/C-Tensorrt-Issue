cmake_minimum_required(VERSION 3.18)
project(TensorrtIssueCheck)

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR C:/TensorRT-8.4.3.1)
endif ()

set(SERIALIZED_ENGINE_FILE "${CMAKE_SOURCE_DIR}/Data/Example.engine")
if (NOT EXISTS ${SERIALIZED_ENGINE_FILE})
    message(FATAL_ERROR "Serialized engine file not found at ${SERIALIZED_ENGINE_FILE}, Please run the python script to generate it")
else ()
    message(STATUS "Serialized engine file found at ${SERIALIZED_ENGINE_FILE}")
endif ()
# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_executable(cpp_version src/main.cpp)
target_include_directories(cpp_version PUBLIC  ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(cpp_version PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
add_custom_command(TARGET cpp_version POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${SERIALIZED_ENGINE_FILE}
        $<TARGET_FILE_DIR:cpp_version>
        COMMENT "Copying ${SERIALIZED_ENGINE_FILE} to target directory bin path")