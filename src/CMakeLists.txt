#configure the compiler options
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 -std=c++11 -fopenmp")

#set required package
##first find PkgConfig
#find_package(PkgConfig)
#find_package(OpenCV REQUIRED)
#find_package(OpenBlas REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(FFTW3 REQUIRED)
#pkg_check_modules(cv opencv)
#include_directories(${cv_INCLUDE_DIRS})
#link_directories(${cv_LIBRARYS})

#set head include directories

#library projects
#add_library(libmnistload STATIC mnistload.h mnistload.cpp)
#target_link_libraries(libmnistload ${OpenCV_LIBS})

#add_library(libmrcparser STATIC mrcparser.h mrcparser.cpp)
#target_link_libraries(libmrcparser ${OpenCV_LIBS})

#add_library(libmrcprocess STATIC mrcprocess.h mrcprocess.cpp)
#target_link_libraries(libmrcprocess ${OpenCV_LIBS})

#add_library(libcml STATIC cml.h cml.cpp)
#target_link_libraries(libcml ${OpenCV_LIBS} libmnistload libmrcparser libmrcprocess)
#executable projects
add_library(cml_nocv STATIC cml_nocv.h cml_nocv.cpp)
target_link_libraries(cml_nocv /usr/lib64/libopenblas.a)
#add_executable(PickerCNN PickerCNN.cpp)
#target_link_libraries(PickerCNN libmnistload libmrcparser libmrcprocess)

#add_executable(cml_trans cml_trans.cpp)
#target_link_libraries(cml_trans libcml libmrcprocess)
#add_executable(cml_real cml_real.cpp)
#target_link_libraries(cml_real libcml libmrcprocess)
#add_library(cmlcl STATIC cml_cl.h cml_cl.cpp)
#arget_link_libraries(cmlcl /usr/lib/libopenblas.a /usr/lib/x86_64-linux-gnu/libOpenCL.so /usr/lib/libclBLAS.so)

#dd_executable(cml_clblas cml_clblas.cpp)
#arget_link_libraries(cml_clblas cmlcl)
#test program
#add_executable(dft dft.cpp)
#target_link_libraries(dft ${OpenCV_LIBS})

#add_executable(cmlt cml_t.cpp)
#target_link_libraries(cmlt libmrcprocess )

#cml_va
#add_executable(cml_va cml_va.cpp)
#target_link_libraries(cml_va libcml)

#add_executable(cml_dcv cml_dcv.cpp)
#target_link_libraries(cml_dcv libcml)
#cml_nocv
#add_executable(cml_clus cml_clus.cpp)
#target_link_libraries(cml_clus cml_nocv)
#add_executable(cml_nonpart cml_nonpart.cpp)
#target_link_libraries(cml_nonpart cml_nocv)
#add_executable(cml_nonpart_fix cml_nonpart_fix.cpp)
#target_link_libraries(cml_nonpart_fix cml_nocv)
add_executable(cml_noise cml_noise.cpp)
target_link_libraries(cml_noise cml_nocv)
add_executable(cml_sort cml_sort.cpp)
target_link_libraries(cml_sort cml_nocv)
#add_executable(test_mpi test_mpi.cpp)
#target_link_libraries(test_mpi ${OpenMP_LIBS})
