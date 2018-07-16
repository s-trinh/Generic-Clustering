cmake_minimum_required(VERSION 2.8)

project(Generic-Clustering)

find_package(VISP REQUIRED)

# set the list of source files
set(test_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_clustering.cpp
)

foreach(cpp ${test_cpp})
  get_filename_component(target ${cpp} NAME_WE)
  include_directories(${VISP_INCLUDE_DIRS})
  add_executable(${target} ${cpp})
  target_link_libraries(${target} ${VISP_LIBRARIES})
endforeach()
