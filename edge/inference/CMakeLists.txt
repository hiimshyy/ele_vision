cmake_minimum_required(VERSION 3.14)
project(smart_cabin_inference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Options ---
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" ON)
option(USE_OPENMP "Enable OpenMP for multi-threaded inference" ON)

# --- Find NCNN ---
# Try to find ncnn installed on system (via apt or from source build)
find_package(ncnn QUIET)
if(NOT ncnn_FOUND)
    message(STATUS "ncnn not found via find_package, building from source...")
    include(FetchContent)
    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn.git
        GIT_TAG 20240820
        GIT_SHALLOW TRUE
    )
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ncnn)
endif()

# --- OpenMP ---
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# --- Inference Library ---
add_library(cabin_inference STATIC
    src/face_detector.cpp
)
target_include_directories(cabin_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cabin_inference PUBLIC ncnn)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cabin_inference PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- Python Bindings ---
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(cabin_inference_py src/python_bindings.cpp)
    target_link_libraries(cabin_inference_py PRIVATE cabin_inference)
    target_include_directories(cabin_inference_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Install to edge/inference/ so Python can find it
    install(TARGETS cabin_inference_py
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/face_recognition/
    )
endif()

# --- Simple test executable ---
add_executable(test_detector src/test_main.cpp)
target_link_libraries(test_detector PRIVATE cabin_inference)
target_include_directories(test_detector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
