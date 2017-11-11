SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 -std=c++11 -fopenmp")
PROJECT("cmake-spr")
add_subdirectory(src)
#add_subdirectory(build)
#SET(cmake-spr_SOURCE_DIR src)
#SET(cmake-spr_BINARY_DIR build)
#nvcc -arch=sm_35 -O2 -rdc=true -lcublas -lcublas_device -lcudadevrt -lcblas -o dcuda cml_dcuda.cu
#nvcc -arch=sm_61 -std=c++11 -lcublas -lopenblas -o gpuvoting cml_gpu_rate_vote.cu

