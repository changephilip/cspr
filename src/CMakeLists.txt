#configure the compiler options
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 -std=c++11")

#set required package
##first find PkgConfig
#find_package(PkgConfig)
find_package(OpenCV REQUIRED)
#pkg_check_modules(cv opencv)
#include_directories(${cv_INCLUDE_DIRS})
#link_directories(${cv_LIBRARYS})

#set head include directories

#library projects
add_library(libmnistload STATIC mnistload.h mnistload.cpp)
target_link_libraries(libmnistload ${OpenCV_LIBS})

add_library(libmrcparser STATIC mrcparser.h mrcparser.cpp)
target_link_libraries(libmrcparser ${OpenCV_LIBS})

add_library(libmrcprocess STATIC mrcprocess.h mrcprocess.cpp)
target_link_libraries(libmrcprocess ${OpenCV_LIBS})

#executable projects

add_executable(PickerCNN PickerCNN.cpp)
target_link_libraries(PickerCNN libmnistload libmrcparser libmrcprocess)


#test program
#add_executable(dft dft.cpp)
#target_link_libraries(dft ${OpenCV_LIBS})

add_executable(cmlt cml_t.cpp)
target_link_libraries(cmlt libmrcprocess)
