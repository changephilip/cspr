
//#include "cml_nocv.h"
//#include "cml_cuda.h"
//#include "cml_cuda.cu"
#include <time.h>
#include <sys/time.h>
//#include <array>>
#include <malloc.h>
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
#include <algorithm>
#include <cuda_runtime.h>

#define L 128 
#define L_power 16384

using namespace std;
typedef struct{
    int x;
    int y;
}cmlncv_tuple;
struct voted
{
    int index;
    int value;
};
typedef struct{
    int x;
    int y;
    float value;
} cml_retstruc;
bool comp(const voted &a,const voted &b)
{
    return a.value<b.value;
}

float MYSUM(int Num,const float *p){
    float re=0.0f;
    int i;
    for(i=0;i<Num;i++){
        re=re+p[i];
    }
    return re;
}

float cvoting(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,float cons2){
    double a,b,c;
//    double two_pi=6.28318530;
    float angleij;
    double cons=180.0/M_PI;
    a = cos((cmlkj-cmlki)*cons2);
    b = cos((cmljk-cmlji)*cons2);
    c = cos((cmlik-cmlij)*cons2);
//    if ((1-c*c)<0 or (1-b*b)<0){
//        printf("error in cvoting\n");
//        exit(EXIT_FAILURE);
//    }
    float t;
    if ((1+2*a*b*c)>a*a+b*b+c*c) {
        t=(a-b*c)/(sqrt(1-b*b)*sqrt(1-c*c));
        if (t<1 and t>-1){
        angleij = acos(t)*cons;
        }
        else if(t>=1){
         angleij=0.0;
        }
        else if (t<=-1){
         angleij=180.0;
        }
    }
    else {angleij=-10.0;}
    return angleij;
}

cmlncv_tuple NCC_value(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
    //        value[i][j] = BNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
		value[i][j]=1.0f;
//            printf("\n000003");
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cmlncv_tuple NCC_value0(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
//            value[i][j] = NCC0(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
		value[i][j]=1.0f;
//            printf("\n000003");
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cmlncv_tuple NCC_Q(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
    float Qci[after_dft_size][4];
    float Qcj[after_dft_size][4];
//#pragma omp parallel for
    for (i=0;i<after_dft_size;i++){
//        Qci[i][0] = cblas_sasum( after_dft_size, &Ci[i*after_dft_size], 1);//sum
        Qci[i][0] = MYSUM(after_dft_size,&Ci[i*after_dft_size]);
        Qci[i][1] = Qci[i][0] / after_dft_size;//mean
        Qci[i][2] = cblas_sdot( after_dft_size, &Ci[i*after_dft_size], 1,&Ci[i*after_dft_size],1);//dot
        Qci[i][3] = sqrt((Qci[i][2] + after_dft_size*Qci[i][1]*Qci[i][1] - 2*Qci[i][0]*Qci[i][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }
//#pragma omp parallel for
    for (j=0;j<after_dft_size;j++){
//        Qcj[j][0] = cblas_sasum( after_dft_size, &Cj[j*after_dft_size], 1);//sum
        Qcj[j][0] = MYSUM(after_dft_size,&Cj[j*after_dft_size]);
        Qcj[j][1] = Qcj[j][0] / after_dft_size;//mean
        Qcj[j][2] = cblas_sdot( after_dft_size, &Cj[j*after_dft_size],1, &Cj[j*after_dft_size],1);//dot
        Qcj[j][3] = sqrt((Qcj[j][2] + after_dft_size*Qcj[j][1]*Qcj[j][1] - 2*Qcj[j][0]*Qcj[j][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }


    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
            //    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
            value[i][j] = (cblas_sdot(after_dft_size,&Ci[i*after_dft_size],1, &Cj[j*after_dft_size],1 )+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cmlncv_tuple NCC_QT(float **Qci,float **Qcj,float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float Qci[after_dft_size][4];
//    float Qcj[after_dft_size][4];

/*
#pragma omp parallel for
    for (i=0;i<after_dft_size;i++){
        Qci[i][0] = cblas_sasum( after_dft_size, &Ci[i*after_dft_size], 1);//sum
        Qci[i][1] = Qci[i][0] / after_dft_size;//mean
        Qci[i][2] = cblas_sdot( after_dft_size, &Ci[i*after_dft_size], 1,&Ci[i*after_dft_size],1);//dot
        Qci[i][3] = sqrt((Qci[i][2] + after_dft_size*Qci[i][0]*Qci[i][0] - 2*Qci[i][0]*Qci[i][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }
#pragma omp parallel for
    for (i=0;i<after_dft_size;i++){
        Qcj[i][0] = cblas_sasum( after_dft_size, &Cj[i*after_dft_size], 1);//sum
        Qcj[i][1] = Qcj[i][0] / after_dft_size;//mean
        Qcj[i][2] = cblas_sdot( after_dft_size, &Cj[i*after_dft_size],1, &Cj[i*after_dft_size],1);//dot
        Qcj[i][3] = sqrt((Qci[i][2] + after_dft_size*Qcj[i][0]*Qcj[i][0] - 2*Qcj[i][0]*Qcj[i][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }
*/
//    printf("see Qci\n");
//    printf("%f",Qci[0][1]);

    //mpi here
    //old code
    /*
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
            //    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
            value[i][j] = (cblas_sdot(after_dft_size,&Ci[i*after_dft_size],1, &Cj[j*after_dft_size],1 )+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
//            printf("%f",value[i][j]);
        }

    }
    */
    //new code,complete it with sgemm
    //float C1[after_dft_size*after_dft_size];
    float C[after_dft_size*after_dft_size];
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,after_dft_size,after_dft_size,after_dft_size,1,Ci,after_dft_size,Cj,after_dft_size,0,C,after_dft_size);

    for (i=0;i<after_dft_size;i++){
//#pragma omp parallel for
        for (j=0;j<after_dft_size;j++){
            value[i][j]=(C[i*after_dft_size+j]+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }
    }





    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
    //to deal with value_ini<0.7,the ncc_value shouldn't be too small
    if (value_ini<0.5){
        ret.x=-1;
        ret.y=-1;
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

float max_float(float *infloat,int size_of_array){
    int i;
    float max_return;
    max_return=infloat[0];
    for (i=1;i<size_of_array;i++){
        if (max_return < infloat[i]){
            max_return=infloat[i];
        }
    }
    return max_return;
}

int max_float_index(float *infloat,int size_of_array){
    int i;
    float max_float;
    int max_index_return;
    max_index_return=0;
    max_float=infloat[0];
    for (i=1;i<size_of_array;i++){
        if (max_float < infloat[i]){
            max_float = infloat[i];
            max_index_return=i;
        }
    }
    return max_index_return;
}
//__device__ float 



__global__ void simple_max_kernel(float *data,float *d_max_value,int *d_max_index){
    float maxvalue=data[0];
    int i;
    int index=0;
    for (i=0;i<L_power;i++){
        if (maxvalue<data[i]){
            maxvalue=data[i];
            index=i;
        }
    }
    *d_max_value=maxvalue;
    *d_max_index=index;
}
__global__ void my_reduction_kernel1(float *d_in,float *d_out,int *d_out_index,int N){
    const int globalid=(blockDim.x*blockIdx.x+threadIdx.x);
    int i;
    float local_max=d_in[globalid];
    int local_max_index=globalid;
    for (int i=1;i<N;i++){
        local_max=fmaxf(local_max,d_in[globalid+i*N]);
        if (local_max==d_in[globalid+i*N]){
            local_max_index=globalid+i*N;
            }
        }
    d_out[threadIdx.x]=local_max;
    d_out_index[threadIdx.x]=local_max_index;
    }
__global__ void my_reduction_kernel2(float *d_in,int *index_in,float *d_out,int *d_out_index,int N){
    float local_max=d_in[0];
    int max_index=index_in[0];
    for (int i=1;i<N;i++){
        local_max=fmaxf(local_max,d_in[i]);
        if (local_max==d_in[i]){
            max_index=index_in[i];
            }
        }
    *d_out=local_max;
    *d_out_index=max_index;
}
__global__ void flambda(float *d_process,float *d_sum,float *d_mean,float *d_stdv,int image_a,int image_b){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    int i=blockIdx.x;
    int j=threadIdx.x;
    d_process[id]=(d_process[id]+L*d_mean[image_a*L+i]*d_mean[image_b*L+j]-d_sum[image_a*L+i]*d_mean[image_b*L+j]-d_mean[image_a*L+i]*d_sum[image_b*L+j])/(L*d_stdv[image_a*L+i]*d_stdv[image_b*L+j]);
}


__global__ void block_kernel(int *d_ctr1,int *d_ctr2,float *d_data,float *d_sum,float *d_mean,float *d_stdv,float *d_Svalue,int *d_max_index){
    //blockid.x,.y->image_a,image_b;

    //do a pair of NCC in a block
    //size=128*128,16 per thread
    //use 1024 thread to calculate
    //16 sdot needed to used first
    //shared memory 1024 float and 1024 int
    __shared__ float sp[1024];
    __shared__ int si[1024];

    int i;
    float threadmax=0.0f;
    int threadindexmax=0;
    float dot_result[16];
    //float dot_result2[16];
    int index_a[16];
    int index_b[16];
    int index[16];
    int globalblockid=blockIdx.x+gridDim.x*blockIdx.y;
    int image_a=d_ctr1[globalblockid];
    int image_b=d_ctr2[globalblockid];
    int globalthread=threadIdx.x+threadIdx.y*32;
//    cublasHandle_t handle;
//    cublasCreate(&handle);
    //threadIdx.x,threadIdx.y
    //dot_result[0]=cublasSdot(handle,L,,1,,1);
    index_a[0]=threadIdx.x*4;
    index_b[0]=threadIdx.y*4;

    index_a[1]=index_a[0]+1;
    index_a[2]=index_a[0]+2;
    index_a[3]=index_a[0]+3;
    index_a[4]=index_a[0];
    index_a[5]=index_a[0]+1;
    index_a[6]=index_a[0]+2;
    index_a[7]=index_a[0]+3;
    index_a[8]=index_a[0];
    index_a[9]=index_a[0]+1;
    index_a[10]=index_a[0]+2;
    index_a[11]=index_a[0]+3;
    index_a[12]=index_a[0];
    index_a[13]=index_a[0]+1;
    index_a[14]=index_a[0]+2;
    index_a[15]=index_a[0]+3;

    index_b[1]=index_b[0];
    index_b[2]=index_b[0];
    index_b[3]=index_b[0];
    index_b[4]=index_b[0]+1;
    index_b[5]=index_b[0]+1;
    index_b[6]=index_b[0]+1;
    index_b[7]=index_b[0]+1;
    index_b[8]=index_b[0]+2;
    index_b[9]=index_b[0]+2;
    index_b[10]=index_b[0]+2;
    index_b[11]=index_b[0]+2;
    index_b[12]=index_b[0]+3;
    index_b[13]=index_b[0]+3;
    index_b[14]=index_b[0]+3;
    index_b[15]=index_b[0]+3;

    index[0]=index_b[0]*L+index_a[0];
    index[1]=index_b[1]*L+index_a[1];
    index[2]=index_b[2]*L+index_a[2];
    index[3]=index_b[3]*L+index_a[3];
    index[4]=index_b[4]*L+index_a[4];
    index[5]=index_b[5]*L+index_a[5];
    index[6]=index_b[6]*L+index_a[6];
    index[7]=index_b[7]*L+index_a[7];
    index[8]=index_b[8]*L+index_a[8];
    index[9]=index_b[9]*L+index_a[9];
    index[10]=index_b[10]*L+index_a[10];
    index[11]=index_b[11]*L+index_a[11];
    index[12]=index_b[12]*L+index_a[12];
    index[13]=index_b[13]*L+index_a[13];
    index[14]=index_b[14]*L+index_a[14];
    index[15]=index_b[15]*L+index_a[15];

//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[0]*L],1,&d_data[image_b*L_power+index_b[0]*L],1,&dot_result[0]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[1]*L],1,&d_data[image_b*L_power+index_b[1]*L],1,&dot_result[1]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[2]*L],1,&d_data[image_b*L_power+index_b[2]*L],1,&dot_result[2]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[3]*L],1,&d_data[image_b*L_power+index_b[3]*L],1,&dot_result[3]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[4]*L],1,&d_data[image_b*L_power+index_b[4]*L],1,&dot_result[4]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[5]*L],1,&d_data[image_b*L_power+index_b[5]*L],1,&dot_result[5]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[6]*L],1,&d_data[image_b*L_power+index_b[6]*L],1,&dot_result[6]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[7]*L],1,&d_data[image_b*L_power+index_b[7]*L],1,&dot_result[7]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[8]*L],1,&d_data[image_b*L_power+index_b[8]*L],1,&dot_result[8]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[9]*L],1,&d_data[image_b*L_power+index_b[9]*L],1,&dot_result[9]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[10]*L],1,&d_data[image_b*L_power+index_b[10]*L],1,&dot_result[10]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[11]*L],1,&d_data[image_b*L_power+index_b[11]*L],1,&dot_result[11]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[12]*L],1,&d_data[image_b*L_power+index_b[12]*L],1,&dot_result[12]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[13]*L],1,&d_data[image_b*L_power+index_b[13]*L],1,&dot_result[13]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[14]*L],1,&d_data[image_b*L_power+index_b[14]*L],1,&dot_result[14]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[15]*L],1,&d_data[image_b*L_power+index_b[15]*L],1,&dot_result[15]);
    for (i=0;i<16;i++){
	dot_result[i]=0.0f;
	    for (int j=0;j<128;j++){
       		 dot_result[i]+=d_data[image_a*L_power+index_a[i]*L+j]*d_data[image_b*L_power+index_b[i]*L+j];
        	}
	}
//    cublasDestroy(handle);
    //flambda
    for (i=0;i<16;i++){
    dot_result[i]+=(L*d_mean[image_a*L+index_a[i]]*d_mean[image_b*L+index_b[i]]-d_sum[image_a*L+index_a[i]]*d_mean[image_b*L+index_b[i]]-d_mean[image_a*L+index_a[i]]*d_sum[image_b*L+index_b[i]]);
	dot_result[i]/=(L*d_stdv[image_a*L+index_a[i]]*d_stdv[image_b*L+index_b[i]]);
   }
//    dot_result[0]=(dot_result[0]+L*d_mean[image_a*L+index_a[0]]*d_mean[image_b*L+index_b[0]]-d_sum[image_a*L+index_a[0]]*d_mean[image_b*L+index_b[0]]-d_mean[image_a*L+index_a[0]]*d_sum[image_b*L+index_b[0]])/(L*d_stdv[image_a*L+index_a[0]]*d_stdv[image_b*L+index_b[0]]);


    //cal the max of 16
    threadmax=dot_result[0];
    threadindexmax=index[0];
    for (i=1;i<16;i++){
        threadmax=fmaxf(threadmax,dot_result[i]);
        if (threadmax==dot_result[i]){
            threadindexmax=index[i];
        }
    }
    sp[globalthread]=threadmax;
    si[globalthread]=threadindexmax;


    __syncthreads();

    for (int activethreads  = 1024 >>1;activethreads>32;activethreads >>=1){
        if (globalthread < activethreads){
            sp[globalthread] = fmaxf(sp[globalthread],sp[globalthread+activethreads]);
            if (sp[globalthread]==sp[globalthread+activethreads]){
                si[globalthread]=si[globalthread+activethreads];
            }
        }
        __syncthreads();
    }

    if (globalthread<32){
        volatile float *ws=sp;
        volatile int *wi=si;
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+32]);
        if (ws[globalthread]==ws[globalthread+32]){
            wi[globalthread]=wi[globalthread+32];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+16]);
        if (ws[globalthread]==ws[globalthread+16]){
            wi[globalthread]=wi[globalthread+16];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+8]);
        if (ws[globalthread]==ws[globalthread+8]){
            wi[globalthread]=wi[globalthread+8];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+4]);
        if (ws[globalthread]==ws[globalthread+4]){
            wi[globalthread]=wi[globalthread+4];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+2]);
        if (ws[globalthread]==ws[globalthread+2]){
            wi[globalthread]=wi[globalthread+2];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+1]);
        if (ws[globalthread]==ws[globalthread+1]){
            wi[globalthread]=wi[globalthread+1];
        }
        if (globalthread==0){
            volatile int *wi=si;
            volatile float *ws=sp;
            d_Svalue[globalblockid]=ws[0];
            d_max_index[globalblockid]=wi[0];
        }
    }




}




void block_wrapper_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
    int c_size=N*(N-1)/2;
    int a,b;
    if (N%2==0){
        a=N-1;
        b=N/2;
    }
    else {
        a=N;
        b=(N-1)/2;
    }

    float *my_sum;
    float *my_mean;
    float *my_stdv;
    int *max_index;

    max_index = new int [c_size];

    my_sum = new float [N*L];
    my_mean = new float [N*L];
    my_stdv = new float [N*L];

    for (int i=0;i<N;i++){
        for (int j=0;j<L;j++){
            my_sum[i*L+j]=help[i][j][0];
            my_mean[i*L+j]=help[i][j][1];
            my_stdv[i*L+j]=help[i][j][3];
        }
    }


    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
//    int *d_Sx;
//    int *d_Sy;

    int *d_max_index;
    float *d_Svalue;

    //float *d_buffer;


    cudaMalloc((void **)&d_data,sizeof(float)*N*L_power);

    cudaMalloc((void **)&d_sum,sizeof(float)*N*L);
    cudaMalloc((void **)&d_mean,sizeof(float)*N*L);
    cudaMalloc((void **)&d_stdv,sizeof(float)*N*L);

//    cudaMalloc((void **)&d_Sx,sizeof(int)*c_size);
//    cudaMalloc((void **)&d_Sy,sizeof(int)*c_size);
    cudaMalloc((void **)&d_max_index,sizeof(int)*c_size);
    cudaMalloc((void **)&d_Svalue,sizeof(float)*c_size);

    cudaMemcpy(d_data,data,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum,my_sum,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,my_mean,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,my_stdv,sizeof(float)*N*L,cudaMemcpyHostToDevice);

//    int ctr_id1[c_size];
//    int ctr_id2[c_size];
    int *ctr_id1;
    int *ctr_id2;
    ctr_id1 = new int [c_size];
    ctr_id2 = new int [c_size];
    for (int i=0;i<N;i++){
        for (int j=i+1;j<N;j++){
            ctr_id1[((2*N-1-i)*i/2+j-i-1)]=i;
            ctr_id2[((2*N-1-i)*i/2+j-i-1)]=j;
        }
    }
    int *d_ctr_id1;
    int *d_ctr_id2;

    cudaMalloc((void **)&d_ctr_id1,sizeof(int)*c_size);
    cudaMalloc((void **)&d_ctr_id2,sizeof(int)*c_size);

    cudaMemcpy(d_ctr_id1,ctr_id1,sizeof(int)*c_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id2,ctr_id2,sizeof(int)*c_size,cudaMemcpyHostToDevice);
    dim3 dimGrid(a,b,1);
    dim3 dimBlock(32,32,1);
    unsigned int sharedsize= 1024*(sizeof(float)+sizeof(int));
    printf("line 1110\n");
    block_kernel<<<dimGrid,dimBlock>>>(d_ctr_id1,d_ctr_id2,d_data,d_sum,d_mean,d_stdv,d_Svalue,d_max_index);

    //for (int i=0;i<b;i++){
//       huge_kernel<<<1,a,0,stream>>>(a*i,d_ctr_id1,d_ctr_id2,d_data,d_sum,d_mean,d_stdv,d_buffer,d_p,d_pi,d_Svalue,d_max_index,L);

//    }
    cudaDeviceSynchronize();
    cudaMemcpy(max_index,d_max_index,sizeof(int)*c_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Svalue,d_Svalue,sizeof(float)*c_size,cudaMemcpyDeviceToHost);
    //cudaStreamDestroy(stream);

    int *ctr_L1;
    int *ctr_L2;
    ctr_L1 = new int [L_power];
    ctr_L2 = new int [L_power];
    for (int i=0;i<L;i++){
        for (int j=0;j<L;j++){
            ctr_L1[i*L+j]=i;
            ctr_L2[i*L+j]=j;
        }
    }
    for(int i=0;i<c_size;i++){
        Sx[i] = ctr_L1[max_index[i]];
        Sy[i] = ctr_L2[max_index[i]];
    }
    delete[] ctr_L1;
    delete[] ctr_L2;
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_max_index);
    cudaFree(d_Svalue);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
    delete[] my_sum;
    delete[] my_mean;
    delete[] my_stdv;
    delete[] ctr_id1;
    delete[] ctr_id2;
    delete[] max_index;

}

int main(int argc ,char* argv[]){
    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */

//    int directreaddisk_flag=0;
    int cml_size=0;
    int N=-1;
    int N_noise=-1;
    int START=1;
    int version=-1;
    int hist_flag=0;
    int iteration=0;
    int debug_flag=0;
    int iteration_SIZE=0;
//    int noise_flag=0;
    char* filename;
    char* good_particle;
    char* logfile;
    char* noisefile;
//    char* directdiskfile;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:n:f:p:v:k:i:d:l:o:hN:")) != -1)
    {
        switch(oc)
        {
        case 's':
            cml_size=atoi(optarg);
            break;
        case 'n':
            N=atoi(optarg);
            break;
        case 'N':
            noisefile=optarg;
            break;
        case 'f':
            filename=optarg;
            break;
        case 'p':
            START=atoi(optarg);
            break;
        case 'v':
            version=atoi(optarg);
            break;
        case 'k':
            hist_flag=atoi(optarg);
            break;
        case 'i':
            iteration=1;
            iteration_SIZE=atoi(optarg);
            break;
        case 'd':
            debug_flag=1;
            break;
        case 'l':
            good_particle=optarg;
            break;
        case 'o':
            logfile=optarg;
            break;
        case 'h':
            printf("CML_NONPART is a program that picks high quality particles and throw out non-particles\n");
            printf("Author:\tQian jiaqiang ,Fudan Univesity,15210700078@fudan.edu.cn\tHuangQiang Lab\n");
            printf("-s the particle size after dft and linear polar\n");
            printf("-n the number of particles\n");
            printf("-N the file contains noise images\n");
            printf("-f the mrcs file which contains the particles data\n");
            printf("-p the start position of particles in mrcs file,default 1\n");
            //printf("-v the calculate method we used,default -1,needn't changed");
            printf("-i if you use -i,then we will try to throw out non-particles\n");
            printf("-l your output filename,which will contain the particles\n");
            printf("example\n");
            printf("cml_noise -s 144 -n 0 -N ~/data/noise.mrc -f ~/data/data.mrc -i -l ~/output/particles -o ~/output/log\n");
            exit(EXIT_SUCCESS);
            break;
        }
    }



    if (!filename) {
        printf("-f filename's abstract path is needed\n");
        exit(EXIT_FAILURE);
    }
    if (!good_particle){
        printf("-l output file name 's abstract path is needed\n");
    }
    if (N==-1 or cml_size==0){
        printf("-n N and -s dft_cml_size are both needed,if N=0,then N=max\n");
        exit(EXIT_FAILURE);
    }
    if (!logfile){
        printf("-o logfile is needed.\n");
        exit(EXIT_FAILURE);
    }
    if (!noisefile){
        printf("-N noise file is needed.\n");
        exit(EXIT_FAILURE);
    }
    FILE *OUTFILE;
    OUTFILE=fopen(logfile,"a+");
    fprintf(OUTFILE,"cml_size\t%d\n",cml_size);
    fprintf(OUTFILE,"N\t%d\n",N);

//全局参数设定
    int dft_size=cml_size;
    int dft_size_pow=dft_size*dft_size;
    int T;
    T=72;
    int i,j,k;
    float sigma=180.0/float(T);
    FILE *f;
    FILE *outputfile;
    FILE *fnoise;
    char mybuf[1024];
    char mybuf2[1024];

//    f=fopen("/home/qjq/data/qjq_200_data","rb");
    f=fopen(filename,"rb");
    fnoise=fopen(noisefile,"rb");

    outputfile=fopen(good_particle,"a+");
    setvbuf(outputfile,mybuf,_IOLBF,1024);
    setvbuf(OUTFILE,mybuf2,_IOLBF,1024);
    long filelength;
    fseek(f,0,SEEK_END);
    filelength=ftell(f);
    if (START-1<0 or (START-1+N)*dft_size_pow>filelength){
        printf("-p START can't L.E 0 or GREATER THAN filelength");
        exit(EXIT_FAILURE);
    }
    rewind(f);
    fprintf(OUTFILE,"length\t%ld\n",filelength);
    if (N==0){
        N=filelength/dft_size_pow/sizeof(float);
    }
    if (N>filelength/dft_size_pow/sizeof(float)){
        fprintf(OUTFILE,"N can't be larger than the max numofItem\n");
        exit(EXIT_FAILURE);
    }

    long noisefilelength;
    fseek(fnoise,0,SEEK_END);
    noisefilelength=ftell(fnoise);
    rewind(fnoise);
    N_noise=noisefilelength/dft_size_pow/sizeof(float);
    int List_Noise[N_noise];
    for (i=0;i<N_noise;i++){
        List_Noise[i]=i;
    }

    int t_start,t_read_file,t_ncc_value,t_end,t_all_end,t_vote_1,t_vote_2,t_vote_3;
    int t_ncc_gpu;
    struct timeval tsBegin,tsEnd;
    t_start=time(NULL);


//    std::vector<int> Global_good_particle;
//    int alpha_ij;
    int iteration_size;
    if (iteration==1){
        //可以选择读取所有粒子数据到硬盘，也可以选择每次单独读取，先选择每次单独读取，节约内存资源
        iteration_size=iteration_SIZE;
        int n_iteration;

        int last_iteration=N%iteration_size;
        if (last_iteration==0){
            n_iteration=N/iteration_size;
        }
        else {
            n_iteration=N/iteration_size+1;
        }
        fprintf(OUTFILE,"n_iteration \t%d\n",n_iteration);
        int control_struct[n_iteration][2];
        //control_struct[local_start][this iteration's particles num]切分数据集，每个数据集为iteration_size
        for (int t=0;t<n_iteration;t++){
            control_struct[t][0]=iteration_size*t;//记录这个划分的第一个粒子的位置，编号-1;
            control_struct[t][1]=iteration_size;//这个子集的粒子数量;
        }
        if (last_iteration!=0){
        control_struct[n_iteration-1][1]=last_iteration;
        }


        int control=0;
        for  (control=0;control<1;control++){//在每次control中，完成iteration_size的计算
            //初始化cml矩阵
            int local_N=control_struct[control][1];//这次control中的粒子数量
            int local_start=control_struct[control][0]*dft_size_pow;//文件中，这次control的粒子的起始位置
            int double_local_N=2*local_N;
//            int local_size_index=local_size*(local_size-1)/2;
            int *cml_pair_matrix[double_local_N];
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix[i]= new int[double_local_N];
                }
	/*
            int *cml_pair_matrix_help[double_local_N];
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix_help[i] = new int[double_local_N];
                }
	*/
                //初始化数据集
            long malloc_size=double_local_N*dft_size_pow;
            float *lineardft_matrix=new float[malloc_size];
            fseek(f,local_start*sizeof(float),SEEK_SET);
            fread(lineardft_matrix,sizeof(float),local_N*dft_size_pow,f);
            //读入Noise文件，将Noise文件乱序，取和local_N等量的噪音图像
            std::random_shuffle(List_Noise,List_Noise+N_noise);
            for (i=0;i<local_N;i++){
                fseek(fnoise,List_Noise[i]*dft_size_pow*sizeof(float),SEEK_SET);
                fread(&lineardft_matrix[(local_N+i)*dft_size_pow],sizeof(float),dft_size_pow,fnoise);
            }
            fprintf(OUTFILE,"read finished\n");

            //维护混合数据List，记录粒子信息,(？不需要维护局部表）
//            int maintain_list[double_local_N];
            //局部粒子矩阵，不需要做粒子数据偏移，但对粒子的序号在输出时要做偏移
            //计算cml_pair_matrix 旧方法,a cblas version
            if (version == 0){
                for (i=0;i<local_N;i++){
                    for (j=0;j<local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
                            tmp=NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            //tmp=CMLNCV::NCC_Q(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            cml_pair_matrix[i][j]=tmp.x;
                            cml_pair_matrix[j][i]=tmp.y;
                        }
                    }
                }
            }
//	    printf("702\n");
            //using a faster version
            if (version == 1){
                for (i=0;i<local_N;i++){
                    for (j=0;j<local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
                            //tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=NCC_Q(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            cml_pair_matrix[i][j]=tmp.x;
                            cml_pair_matrix[j][i]=tmp.y;
                        }
                    }
                }
            }
		printf("703\n");
            t_start=time(NULL);
            //the fastest cpu version
            if (version == -1){
            //cml_pair_matrix 新方法
                float **total_nccq[double_local_N];
            //    float ;
            //    total_nccq = new float** [N];
                int postion;
                for (i=0;i<double_local_N;i++){
                    total_nccq[i] = new float* [dft_size];
                }
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<dft_size;j++){
                        total_nccq[i][j] = new float[4];
                    }
                }
                printf("000005\n");
//                gettimeofday(&tsBegin,NULL);
                for (i=0;i<double_local_N;i++){
            //        #pragma omp parallel for,can't openmp here
                    for (j=0;j<dft_size;j++){
                        postion=i*dft_size_pow+j*dft_size;
            //            printf("000006\n");
//                        total_nccq[i][j][0] = cblas_sasum( dft_size, &lineardft_matrix[postion], 1);//sum
                        total_nccq[i][j][0] = MYSUM(dft_size,&lineardft_matrix[postion]);//sum
            //            printf("000007\n");
                        total_nccq[i][j][1] = total_nccq[i][j][0] / dft_size;//mean
            //            printf("000008\n");
                        total_nccq[i][j][2] = cblas_sdot( dft_size, &lineardft_matrix[postion], 1,&lineardft_matrix[postion],1);//dot
            //            printf("000009\n");
                        total_nccq[i][j][3] = sqrt((total_nccq[i][j][2] + dft_size*total_nccq[i][j][1]*total_nccq[i][j][1] - 2*total_nccq[i][j][0]*total_nccq[i][j][1])/dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
                    }
                }

	/*
//calculate ncc with cpu
                for (i=0;i<double_local_N;i++){
                #pragma omp parallel for
                    for (j=i+1;j<double_local_N;j++){
//                    for (j=i+1;j<double_local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
            //                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=NCC_QT(total_nccq[i],total_nccq[j],&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            cml_pair_matrix[i][j]=tmp.x;
                            cml_pair_matrix[j][i]=tmp.y;
                        }
                    }
                }

	*/
//calculate ncc with gpu
//        cml_retstruc *S;
        //
        int *Sx;
        int *Sy;
        float *Svalue;
        Sx= new int [double_local_N*(double_local_N-1)/2];
        Sy= new int [double_local_N*(double_local_N-1)/2];
        Svalue= new float [double_local_N*(double_local_N-1)/2];
//                S = new cml_retstruc[double_local_N*(double_local_N-1)/2];
		printf("before enter wrappper\n");
		t_ncc_gpu=time(NULL);
       //         wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,Sx,Sy,Svalue);
//		stream_wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,Sx,Sy,Svalue);
//                huge_wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,Sx,Sy,Svalue);
                block_wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,Sx,Sy,Svalue);
                for (i=0;i<double_local_N;i++){
                    for (j=i+1;j<double_local_N;j++){
                        if (Svalue[(2*double_local_N-1-i)*i/2+j-(i+1)]>0.5){
//                        cml_pair_matrix_help[i][j]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].x;
//                        cml_pair_matrix_help[j][i]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].y;
                        cml_pair_matrix[i][j]=Sx[(2*double_local_N-1-i)*i/2+j-(i+1)];
                        cml_pair_matrix[j][i]=Sy[(2*double_local_N-1-i)*i/2+j-(i+1)];
                        }

                    else{
                        cml_pair_matrix[i][j]=-1;
                        cml_pair_matrix[j][i]=-1;
                    }}
                }
/*
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix[i][i]=-1;
                    cml_pair_matrix_help[i][i]=-1;
                }
*/
                //test GPU with cpu
/*
                float diff=0.0f;
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<double_local_N;j++){
                        diff+=(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j])*(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j]);
                        printf("%d\t%d\t%d\t%d\n",i,j,cml_pair_matrix[i][j],cml_pair_matrix_help[i][j]);
                    }
                }
*/
		//for (i=0;i<double_local_N;i++){
		//	for (j=0;j<double_local_N;j++){
		//	printf("%d\t",cml_pair_matrix_help[i][j]);}
		//	printf("\n");
//}
//		for (i=0;i<double_local_N*(double_local_N-1)/2;i++){
//			printf("%d\t%d\t%d\t%f\n",i,Sx[i],Sy[i],Svalue[i]);
//		}
//		printf("\n");
//                diff=sqrt(diff/double_local_N/double_local_N);
//                printf("diff between gpu_ncc with cpu_ncc\t%f\n",diff);
		delete[] Sx;
		delete[] Sy;
		delete[] Svalue;
                //test
                /*
                for (i=0;i<double_local_N;i++){
//                #pragma omp parallel for
                    for (j=0;j<double_local_N;j++){
//                    for (j=i+1;j<double_local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
            //                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=CMLNCV::NCC_QT(total_nccq[i],total_nccq[j],&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            if (cml_pair_matrix[i][j]!=tmp.x) printf("error\n");
                            if (cml_pair_matrix[j][i]!=tmp.y) printf("error\n");
                        }
                    }
                }
                */
                //NCC计算完成，所有common line被算出，释放计算辅助的数据存储矩阵
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<dft_size;j++){
                        delete[] total_nccq[i][j];
                        }
                    }
                for (i=0;i<double_local_N;i++){
                        delete[] total_nccq[i];
                    }

            }
            fprintf(OUTFILE,"ncc cal finished\n");
            t_ncc_value=time(NULL);

            float *hist_peak =  new float[double_local_N*(double_local_N-1)/2];
            int *hist_index = new int[double_local_N*(double_local_N-1)/2];
            float half_pow_pi=sqrt(2*M_PI)*sigma;
//            float four_sigma_pow=4*sigma*sigma;
            float two_sigma_pow=2*sigma*sigma;
//            float alpha_t_alpha12;
            float cons=2*M_PI/dft_size;
            float Trecurse=180.0/T;
            //开始voting算法，先计算一遍voting，算出hist数组、hist_index数组

            //combine the voting and peak
//            for (i=0;i<local_N;i++){
//                        for (j=i+1;j<local_N;j++){
//                            //be sure ij,ji!=-1;

//                            alpha_ij=((2*local_N-1-i)*i/2+j-(i+1));
//                            float tmp_voting[local_N];
//                            float tmp_hist[72]={0.0};
//                            #pragma omp parallel for
//                            for (k=0;k<local_N;k++){
//                                //....#
//                //                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N+k;this is the error that caused difference between cml_dcv and cml_va
//                                if (k!=i and k!=j and cml_pair_matrix[i][j]>-1 and cml_pair_matrix[i][k]>-1 and cml_pair_matrix[j][i]>-1 and cml_pair_matrix[j][k]>-1 and cml_pair_matrix[k][i]>-1 and cml_pair_matrix[k][j]>-1) {
//                                    tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
//                                }
//                                else {
//                                    tmp_voting[k]=-10.0;
//                                }
//                            }

//                            for (int m=0;m<local_N;m++){

//                                    float tmp=tmp_voting[m];
//                                    if (tmp!=-10.0 and tmp!=-9.0){
//            #pragma omp parallel for
//                                    for (int l=0;l<T;l++){
//                                        float alpha_t_alpha12=(180.0*l/float(T))-tmp;
//                                        tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/(four_sigma_pow))/half_pow_pi;
//                                    }
//                                }
//                            }
//                            hist_peak[alpha_ij]=CMLNCV::max_float(tmp_hist,T);
//                           hist_index[alpha_ij]=CMLNCV::max_float_index(tmp_hist,T);

//                        }
//                    }


            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    long int alpha_ij=((2*double_local_N-1-i)*i/2+j-(i+1));
                    if (cml_pair_matrix[i][j]!=-1){
                        float tmp_voting[double_local_N];
                        float tmp_hist[72]={0.0};
#pragma omp parallel for
                        for (k=0;k<double_local_N;k++){
                            if (k!=i and k!=j){
                                if (cml_pair_matrix[i][k]!=-1 and cml_pair_matrix[j][k]!=-1){
                                    tmp_voting[k]=cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
                                }
                                else {tmp_voting[k]=-10.0;}
                            }
                            else {tmp_voting[k]=-10.0;}
                        }
                        for (int m=0;m<double_local_N;m++){
                            float tmp=tmp_voting[m];
                            if (tmp!=-10.0){
#pragma omp parallel for
                                for (int l=0;l<T;l++){
                                    float alpha_t_alpha12=Trecurse*l-tmp;
                                    tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/two_sigma_pow)/half_pow_pi;
                                }
                            }
                        }
                        hist_peak[alpha_ij]=max_float(tmp_hist,T);
                    }
                    else{
                    hist_peak[alpha_ij]=0.0;
                    }
                }
            }
            t_vote_1=time(NULL);
            if (hist_flag){
            fprintf(outputfile,"alpha_ij\ti\tj\thist_index\tpeak_value\n");
            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    int index=(2*double_local_N-1-i)*i/2+j-(i+1);
                    fprintf(outputfile,"alpha_ij\t%d\t%d\t%d\t%f\n",i,j,hist_index[index],hist_peak[index]);
                }
            }
            }
            fprintf(OUTFILE,"\n311\n");
            //从CML_Pair中找出优秀粒子保留，等于剔除non-particle
            //计算每一个alphaij的voting序号
            int NumOfHighPeak=0;

            //固定threshold值，从右侧开始取第一个峰值,先固定threshold
            //threshold 为r/sqrt（local_N),r为2、4、8等。排序取local_N*threshold个为符合条件者。
//            float max_hist_peak=CMLNCV::max_float(hist_peak,local_N*(local_N-1)/2);
            int r=4;

            //复制一个hist_peak的备份;
            float *hist_peak_cp =  new float[double_local_N*(double_local_N-1)/2];
            long hist_peak_size=double_local_N*(double_local_N-1)/2;
            int threshold_top=floor((1-(r/sqrt(double_local_N)))*hist_peak_size);
            for (i=0;i<hist_peak_size;i++){
                hist_peak_cp[i]=hist_peak[i];
            }
            sort(hist_peak_cp,hist_peak_cp+hist_peak_size);
            fprintf(OUTFILE,"threshold_top %d\n",threshold_top);
            float threshold=hist_peak_cp[threshold_top-1];
            delete[] hist_peak_cp;
            t_vote_2=time(NULL);
            int Result_voted[double_local_N];
            for (i=0;i<double_local_N;i++){
                Result_voted[i]=0;
            }
            //找出Peak值较高的CML_pair
            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    int index = (2*double_local_N-1-i)*i/2+j-(i+1);
                    if (hist_peak[index]>threshold) {
//                        NumOfHighPeak = NumOfHighPeak +1;
                        Result_voted[i]=Result_voted[i]+1;
                        Result_voted[j]=Result_voted[j]+1;
                        }
                    }
                }
            float mean_noise=0.0;
            for (i=local_N;i<double_local_N;i++){
                mean_noise+=Result_voted[i];

            }
            mean_noise=mean_noise/local_N;
            fprintf(OUTFILE,"mean_noise\t%f\n",mean_noise);
            //排序result_voted
            std::vector<voted> V;
            for (i=0;i<double_local_N;i++){
                voted s;
                s.index=i;
                s.value=Result_voted[i];
                V.push_back(s);
            }
            std::sort(V.begin(),V.end(),comp);
            //输出排序值和结果,分两个，noise排位的平均值;随着排位上升，noise的比例
            for (i=0;i<double_local_N;i++){
                fprintf(outputfile,"%d\t%d\t%d\n",i,V[i].index,V[i].value);
            }
            int counter=0;

            for (i=0;i<double_local_N;i++){
                if (V[i].index>=local_N){
                    counter=counter+1;
                }
                if ((i+1)%100==0){
                    fprintf(OUTFILE,"%d\tcounter %d\trate\t%f\n",i,counter,counter/float(i+1));
                }
            }

            //取出高可信粒子
            /*
            const int step=sizeof(Result_voted)/sizeof(int);
            float threshold_voted=(*std::max_element(Result_voted,Result_voted+step))*0.2;
            fprintf(OUTFILE,"threshold_voted is %f\n",threshold_voted);
//            float threshold_voting=*std::max_element(Result_voting,Reslut_voting+step)*0.8;
            //only voted particles are trusted.
            std::vector<int> local_good_particle;
            for (i=0;i<double_local_N;i++){
                if (Result_voted[i]>threshold_voted){
                    int good_voted_particle=control_struct[control][0]+i;
                    local_good_particle.push_back(good_voted_particle);
                }
            }



            if (local_good_particle.size()==0){
                fprintf(outputfile,"the local_good_particle num is 0\n");
            }
            for (auto m:local_good_particle){
                fprintf(outputfile,"%d\n",m);
            }
            */
            //放入全局Global_good_particle中
//            for (auto m:local_good_particle){
//                Global_good_particle.push_back(m);
//            }

            //销毁所有堆
            delete[] lineardft_matrix;
            delete[] hist_index;
            delete[] hist_peak;
            for (i=0;i<double_local_N;i++){
                delete[] cml_pair_matrix[i];
//                delete[] cml_pair_matrix_help[i];
            }
            t_end=time(NULL);
            fprintf(OUTFILE,"ncc_time %d\n",t_ncc_value-t_start);
	    fprintf(OUTFILE,"ncc_gpu %d\n",t_ncc_value-t_ncc_gpu);
            fprintf(OUTFILE,"voting time %d\n",t_end-t_ncc_value);
            fprintf(OUTFILE,"only voting time %d\n",t_vote_1-t_ncc_value);
            fprintf(OUTFILE,"sort time %d\n",t_vote_2-t_vote_1);
            fprintf(OUTFILE,"%d/%d\tcompleted\n",control,n_iteration);

        }

}

        fclose(f);
        fclose(outputfile);
        fclose(fnoise);
        fclose(OUTFILE);






        t_all_end=time(NULL);
        printf("all time %dhour\n",(t_all_end-t_start)/3600);

        return 0;
}
