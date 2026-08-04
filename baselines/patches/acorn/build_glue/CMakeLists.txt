cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(faiss_acorn_glue VERSION 1.7.3 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

if(NOT ACORN_ROOT)
  message(FATAL_ERROR "pass -DACORN_ROOT=<abs path to vendored ACORN>")
endif()
list(APPEND CMAKE_MODULE_PATH "${ACORN_ROOT}/cmake")

set(FAISS_ENABLE_GPU    OFF        CACHE BOOL   "" FORCE)
set(FAISS_ENABLE_PYTHON OFF        CACHE BOOL   "" FORCE)
set(FAISS_ENABLE_C_API  OFF        CACHE BOOL   "" FORCE)
set(FAISS_OPT_LEVEL     "generic"  CACHE STRING "" FORCE)
set(BUILD_TESTING       OFF        CACHE BOOL   "" FORCE)
set(BUILD_SHARED_LIBS   OFF        CACHE BOOL   "" FORCE)

add_subdirectory(${ACORN_ROOT}/faiss ${CMAKE_BINARY_DIR}/faiss_lib)

target_include_directories(faiss PUBLIC ${ACORN_ROOT})

find_package(ZLIB REQUIRED)
target_link_libraries(faiss PUBLIC ZLIB::ZLIB)
