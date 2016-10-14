#ifndef MRCPROCESS_H
#define MRCPROCESS_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv/cv.hpp>
#include <opencv2/core/core_c.h>
#include <opencv/highgui.h>
#define mrcsize 1024

struct _mrchead{
    int nx;
    int ny;
    int nz;
    float mode;
} ;

/*
 class MrcProcess
{

public:
    //MrcProcess();


};
*/
namespace MrcProcess {

struct _mrchead readhead(FILE *f);
void readmrcdata(FILE *f,float* dataperfile,struct _mrchead mrchead);
 cv::Mat mynorm(const cv::Mat &image);
 int showimagecpp(cv::Mat image);
 cv::Mat Gauss_Equal_Norm(cv::Mat image);
 cv::Mat mrcOrigin(FILE *f);
}
#endif // MRCPROCESS_H
