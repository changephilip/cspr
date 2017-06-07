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

__global__ void test_block_kernel(int *d_ctr1,int *d_ctr2,float *d_data,float *d_sum,float *d_mean,float *d_stdv,float *d_Svalue,int *d_max_index){
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
    int index_a[16];
    int index_b[16];
    int index[16];
    int globalblockid=blockIdx.x+gridDim.y*blockIdx.y;
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
//    for (i=0;i<16;i++){
//    dot_result[i]=(dot_result[i]+L*d_mean[image_a*L+index_a[i]]*d_mean[image_b*L+index_b[i]]-d_sum[image_a*L+index_a[i]]*d_mean[image_b*L+index_b[i]]-d_mean[image_a*L+index_a[i]]*d_sum[image_b*L+index_b[i]])/(L*d_stdv[image_a*L+index_a[i]]*d_stdv[image_b*L+index_b[i]]);
//   }
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

int main(){
    
}
