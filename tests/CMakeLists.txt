add_executable(hnsw_stats_tests
  scaffold_test.cpp
  search_test.cpp
  region_test.cpp
)
target_link_libraries(hnsw_stats_tests PRIVATE
  hnsw_stats_core
  hnsw_stats_harness
  Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(hnsw_stats_tests)
