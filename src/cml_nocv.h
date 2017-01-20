#ifndef CML_NOCV_H
#define CML_NOCV_H
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <cblas.h>
using namespace std;
typedef struct{
    int x;
    int y;
}cmlncv_tuple;
namespace CMLNCV
{
    void readDFTData(FILE *f,float *p,float CML_SIZE,int numOfItem);
    float NCC0(float *cml1,float *cml2,int CML_SIZE);
    float BNCC(const float *cml1,const float *cml2,int CML_SIZE);
    float cal_angle(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);
    bool voting_condition(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);
    float MYSUM(int Num,const float *p);
    cmlncv_tuple NCC_value(float *Ci,float *Cj,int after_dft_size);
    cmlncv_tuple NCC_valuet(float *Ci,float *Cj,int after_dft_size);
    cmlncv_tuple NCC_Q(float *Ci,float *Cj,int after_dft_size);
    cmlncv_tuple NCC_QT(float **Qci,float **Qcj,float *Ci,float *Cj,int after_dft_size);
    cmlncv_tuple NCC_value0(float *Ci,float *Cj,int after_dft_size);
    float max_float(float *infloat,int size_of_array);
    int max_float_index(float *infloat,int size_of_array);
    float cvoting(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size);

}
#endif // CML_NOCV_H
