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
//#include <cublas_v2.h>
#define L 128 
#define L_power 16384

__global__ void test_block_kernel(float *d_data,float *d_Svalue,int *d_max_index){
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
    int globalblockid=0;
//    int image_a=d_ctr1[globalblockid];
//    int image_b=d_ctr2[globalblockid];
    int image_a=0;
    int image_b=0;
    int globalthread=threadIdx.x+threadIdx.y*32;
//    cublasHandle_t handle;
//    cublasCreate(&handle);
//    cublasStatus_t status;
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

//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[0]*L],1,&d_data[image_b*L_power+index_b[0]*L],1,&dot_result[0]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[1]*L],1,&d_data[image_b*L_power+index_b[1]*L],1,&dot_result[1]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[2]*L],1,&d_data[image_b*L_power+index_b[2]*L],1,&dot_result[2]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[3]*L],1,&d_data[image_b*L_power+index_b[3]*L],1,&dot_result[3]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[4]*L],1,&d_data[image_b*L_power+index_b[4]*L],1,&dot_result[4]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[5]*L],1,&d_data[image_b*L_power+index_b[5]*L],1,&dot_result[5]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[6]*L],1,&d_data[image_b*L_power+index_b[6]*L],1,&dot_result[6]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[7]*L],1,&d_data[image_b*L_power+index_b[7]*L],1,&dot_result[7]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[8]*L],1,&d_data[image_b*L_power+index_b[8]*L],1,&dot_result[8]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[9]*L],1,&d_data[image_b*L_power+index_b[9]*L],1,&dot_result[9]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[10]*L],1,&d_data[image_b*L_power+index_b[10]*L],1,&dot_result[10]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[11]*L],1,&d_data[image_b*L_power+index_b[11]*L],1,&dot_result[11]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[12]*L],1,&d_data[image_b*L_power+index_b[12]*L],1,&dot_result[12]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[13]*L],1,&d_data[image_b*L_power+index_b[13]*L],1,&dot_result[13]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[14]*L],1,&d_data[image_b*L_power+index_b[14]*L],1,&dot_result[14]);
//    status=cublasSdot(handle,L,&d_data[image_a*L_power+index_a[15]*L],1,&d_data[image_b*L_power+index_b[15]*L],1,&dot_result[15]);
//    float tmp_a[L];
//    float tmp_b[L];
//    for (i=0;i<16;i++){
//	dot_result[i]=0.0f;
//	for (int j=0;j<128;j++){
//		tmp_a[j]=d_data[image_a*L_power+index_a[i]*L+j];
//		}
//	for (int j=0;j<128;j++){
//		tmp_b[j]=d_data[image_b*L_power+index_b[i]*L+j];
//		}
//	for (int j=0;j<128;j++){
//		dot_result[i]+=tmp_a[j]*tmp_b[j];
//		}
//	}
    for (i=0;i<16;i++){
	dot_result[i]=0.0f;
	    for (int j=0;j<128;j++){
       		 dot_result[i]+=d_data[image_a*L_power+index_a[i]*L+j]*d_data[image_b*L_power+index_b[i]*L+j];
//		dot_result[i]=fmaf(d_data[image_a*L_power+index_a[i]*L+j],d_data[image_b*L_power+index_b[i]*L+j],dot_result[i]);

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
    float A[L_power];
    for (int i=0;i<L_power;i++){
        A[i]=i;
    }
    float S_value;
    int max_index;
    float *d_A;
    float *d_sv;
    int  *d_index;
    printf("208\n");
    cudaMalloc((void **)&d_A,sizeof(float)*L_power);
    cudaMalloc((void **)&d_sv,sizeof(float));
    cudaMalloc((void **)&d_index,sizeof(int));
    cudaMemcpy(d_A,A,sizeof(float)*L_power,cudaMemcpyHostToDevice);
    dim3 dimBlock(32,32,1);
    printf("209\n");
    test_block_kernel<<<1,dimBlock>>>(d_A,d_sv,d_index);
    cudaDeviceSynchronize();
    printf("217\n");
    cudaMemcpy(&S_value,d_sv,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_index,d_index,sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_sv);
    cudaFree(d_index);
    printf("224\n");
    printf("svalue\t%f\n",S_value);
    printf("maxindex\t%d\n",max_index);

    return 0;


}
