add_library(hnsw_stats_core STATIC
  hnsw_stats/version.cpp
  hnsw_stats/predicate/ColumnPredicate.cpp
  hnsw_stats/hnsw/HnswIndex.cpp
  hnsw_stats/search/HnswSearch.cpp
  hnsw_stats/search/OursSearch.cpp
  hnsw_stats/region/HubVoronoiRegionBuilder.cpp
  hnsw_stats/region/ForestRegionBuilder.cpp
  hnsw_stats/stats/RegionHistogramStats.cpp
  hnsw_stats/cost_model/EstCostModel.cpp
)
target_include_directories(hnsw_stats_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hnsw_stats_core PUBLIC hnsw_stats_deps)

if(HNSW_STATS_SIMD)
  target_compile_options(hnsw_stats_core PRIVATE -mavx2 -mfma)
endif()
