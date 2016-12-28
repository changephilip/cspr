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
#include "mrcparser.h"
#include "opencv2/imgproc/imgproc_c.h"
using namespace std;
typedef struct {
    int x;
    int y;
}cml_tuple;
struct coordinate{
            int x;
            int y;
        };
typedef map<string,vector<coordinate>> fileNameToCoodinateList;
typedef vector<cv::Mat> imagebuffer;
namespace CML {
    void getsubmrc(float *p,int x,int y,int CML_SIZE,float *s,const struct _mrchead header);
    cv::Mat imdft(cv::Mat &I);
    float NCC0(float *cml1,float *cml2,int CML_SIZE);
    float cal_angle(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);
    bool voting_condition(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);
    cml_tuple NCC_value(float *Ci,float *Cj,int after_dft_size);
    cml_tuple NCC_valuet(float *Ci,float *Cj,int after_dft_size);
//    void voting_algo(int *hist,float *p,int projection_i,int projection_j,int VA_length);
    void cml_read(float *data,fileNameToCoodinateList intable,int cml_size);
    void cml_dftread(float *data, fileNameToCoodinateList intable, int cml_size, int dft_size, int N);
    float max_float(float *infloat,int size_of_array);
    int max_float_index(float *infloat,int size_of_array);
    void linearpolar(cv::Mat &I,cv::Mat &dst);
    void image_to_mat(cv::Mat &I,float *matp,int after_dft_size);
    fileNameToCoodinateList CNNpicker(ifstream &file);
    int calMnistItem(fileNameToCoodinateList intable);
    void bufferWrite(fileNameToCoodinateList intable,FILE *mnistfile,FILE* labellog,int sub_size);
    void writeDisk(float *p,FILE *filename,long filelength);

}

#endif // CML_H
