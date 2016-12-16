SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 -std=c++11 -fopenmp")
PROJECT("cmake-spr")
add_subdirectory(src)
#add_subdirectory(build)
#SET(cmake-spr_SOURCE_DIR src)
#SET(cmake-spr_BINARY_DIR build)

