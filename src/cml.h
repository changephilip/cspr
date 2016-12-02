#ifndef CML_H
#define CML_H
#include "mrcparser.h"
#include "mnistload.h"
#include "mrcprocess.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv/cv.hpp>
#include <opencv2/core/core_c.h>
#include <opencv/highgui.h>
#include <math.h>
#include "opencv2/imgproc/imgproc_c.h"
typedef struct {
    int x;
    int y;
}tuple;

namespace CML {
    void getsubmrc(float *p,int x,int y,int CML_SIZE,float *s,const struct _mrchead header);
    cv::Mat imdft(cv::Mat &I);
    float NCC0(float *cml1,float *cml2,int CML_SIZE);
    float cal_angle(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);
    bool voting_condition(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);
    tuple NCC_value(float *Ci,float *Cj,int after_dft_size);
    void voting_algo(int *hist,float *p,int projection_i,int projection_j,int VA_length);

}

#endif // CML_H
