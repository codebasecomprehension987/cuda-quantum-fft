cmake_minimum_required(VERSION 3.18)
project(cuda_quantum_fft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=64")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library source files
set(QFFT_SOURCES
    src/context.cu
    src/radix8_kernel.cu
    src/bit_reverse.cu
)

# Build shared library
add_library(qfft SHARED ${QFFT_SOURCES})

target_include_directories(qfft PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(qfft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Examples
add_executable(audio_fft examples/audio_fft.cu)
target_link_libraries(audio_fft qfft)

# Benchmarks
add_executable(benchmark_fft examples/benchmark.cu)
target_link_libraries(benchmark_fft qfft)

# Install
install(TARGETS qfft
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.cuh"
)

# Print config
message(STATUS "CUDA Quantum FFT Configuration:")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler flags: ${CMAKE_CUDA_FLAGS}")
