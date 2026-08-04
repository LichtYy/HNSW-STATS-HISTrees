cmake_minimum_required(VERSION 3.24)

project(hnsw_stats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(HNSW_STATS_SIMD "Allow SIMD in our distance kernels (group2). OFF = group1 scalar-only." ON)

find_package(hnswlib CONFIG REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(hnsw_stats_deps INTERFACE)
target_link_libraries(hnsw_stats_deps INTERFACE
  hnswlib::hnswlib
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  Threads::Threads
)
target_compile_features(hnsw_stats_deps INTERFACE cxx_std_20)
if(NOT HNSW_STATS_SIMD)
  target_compile_definitions(hnsw_stats_deps INTERFACE HNSW_STATS_SCALAR_ONLY=1)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/harness)

add_subdirectory(src)
add_subdirectory(harness)

enable_testing()
add_subdirectory(tests)
