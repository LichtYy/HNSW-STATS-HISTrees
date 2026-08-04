add_library(hnsw_stats_harness STATIC
  harness_stub.cpp
  common/Fvecs.cpp
  common/DatasetLoader.cpp
  common/AttributesLoader.cpp
  common/SyntheticColumns.cpp
  common/ClusterField.cpp
  common/RhoRank.cpp
  common/QueryPredicateGen.cpp
  gt/BruteForceGT.cpp
  metrics/MetricsCollector.cpp
  method/OursMethod.cpp
  ${CMAKE_SOURCE_DIR}/baselines/reimpl/post_filter/PostFilter.cpp
  ${CMAKE_SOURCE_DIR}/baselines/reimpl/navix/NavixMethod.cpp
)
target_include_directories(hnsw_stats_harness PUBLIC
  ${CMAKE_SOURCE_DIR}/harness
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/baselines/reimpl
)
target_link_libraries(hnsw_stats_harness PUBLIC hnsw_stats_core hnsw_stats_deps)

if(HNSW_STATS_SIMD)
  target_compile_options(hnsw_stats_harness PRIVATE -mavx2 -mfma)
endif()

set(SERF_DIR ${CMAKE_SOURCE_DIR}/baselines/_vendor/FANNBench/SeRF)
if(EXISTS ${SERF_DIR}/src/segment_graph_1d.h)
  add_library(serf_adapter STATIC
    method/serf/SerfMethod.cpp
    ${SERF_DIR}/src/common/utils.cc)
  target_include_directories(serf_adapter PRIVATE
    ${CMAKE_SOURCE_DIR}/harness ${CMAKE_SOURCE_DIR}/src
    ${SERF_DIR}/src ${SERF_DIR}/src/common ${SERF_DIR}/include ${SERF_DIR}/include/common)
  find_package(OpenMP)
  target_link_libraries(serf_adapter PUBLIC hnsw_stats_core hnsw_stats_deps)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(serf_adapter PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(hnsw_stats_harness PUBLIC serf_adapter)
  target_compile_definitions(hnsw_stats_harness PUBLIC HS_HAVE_SERF=1)
endif()

set(NHQ_DIR ${CMAKE_SOURCE_DIR}/baselines/_vendor/NHQ/NHQ-NPG_kgraph)
if(EXISTS ${NHQ_DIR}/src/index_graph.cpp)
  add_library(nhq_adapter STATIC
    method/nhq/NhqMethod.cpp
    ${NHQ_DIR}/src/index.cpp ${NHQ_DIR}/src/index_graph.cpp
    ${NHQ_DIR}/src/index_kdtree.cpp ${NHQ_DIR}/src/index_random.cpp)
  target_include_directories(nhq_adapter PRIVATE
    ${CMAKE_SOURCE_DIR}/harness ${CMAKE_SOURCE_DIR}/src ${NHQ_DIR}/include)
  target_compile_options(nhq_adapter PRIVATE -mavx2 -mfma -fopenmp)
  target_link_libraries(nhq_adapter PUBLIC hnsw_stats_core hnsw_stats_deps)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nhq_adapter PUBLIC OpenMP::OpenMP_CXX)
  endif()
  find_library(OPENBLAS_LIB NAMES openblas)
  if(OPENBLAS_LIB)
    target_link_libraries(nhq_adapter PUBLIC ${OPENBLAS_LIB})
    target_link_libraries(hnsw_stats_harness PUBLIC nhq_adapter)
    target_compile_definitions(hnsw_stats_harness PUBLIC HS_HAVE_NHQ=1)
  endif()
endif()

set(ACORN_DIR ${CMAKE_SOURCE_DIR}/baselines/_vendor/FANNBench/ACORN)
if(EXISTS ${ACORN_DIR}/build/faiss_lib/libfaiss.a)
  add_library(acorn_adapter STATIC method/acorn_orig/AcornOrigMethod.cpp)
  target_include_directories(acorn_adapter PRIVATE
    ${CMAKE_SOURCE_DIR}/harness ${CMAKE_SOURCE_DIR}/src ${ACORN_DIR})
  target_link_libraries(acorn_adapter PUBLIC hnsw_stats_core hnsw_stats_deps
    ${ACORN_DIR}/build/faiss_lib/libfaiss.a)
  find_library(OPENBLAS_LIB_ACORN NAMES openblas)
  if(OPENBLAS_LIB_ACORN)
    target_link_libraries(acorn_adapter PUBLIC ${OPENBLAS_LIB_ACORN} z m dl)
  endif()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(acorn_adapter PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(hnsw_stats_harness PUBLIC acorn_adapter)
  target_compile_definitions(hnsw_stats_harness PUBLIC HS_HAVE_ACORN=1)
endif()

set(DA_DIR ${CMAKE_SOURCE_DIR}/baselines/_vendor/FANNBench/DiskANN)
if(EXISTS ${DA_DIR}/build/src/libdiskann.a)
  add_library(diskann_adapter STATIC method/filtered_diskann/DiskannMethod.cpp)
  target_include_directories(diskann_adapter PRIVATE
    ${CMAKE_SOURCE_DIR}/harness ${CMAKE_SOURCE_DIR}/src ${DA_DIR}/include)
  target_compile_options(diskann_adapter PRIVATE -mavx2 -mfma -msse2 -fopenmp -DUSE_AVX2 -DMKL_ILP64)
  target_link_libraries(diskann_adapter PUBLIC hnsw_stats_core hnsw_stats_deps
    ${DA_DIR}/build/src/libdiskann.a
    mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl boost_program_options)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(diskann_adapter PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(hnsw_stats_harness PUBLIC diskann_adapter)
  target_compile_definitions(hnsw_stats_harness PUBLIC HS_HAVE_DISKANN=1)
endif()

