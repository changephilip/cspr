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
#include <cublas_v2.h>
#define L 140
#define L_power 19600
/*
 *this program is used to test cuda-function
 * 1.can cublas in kernel read data correctly?
 * 2.can cublas in kernel compute correctly?
 * 3.the data returened by kernel is correctly?
 *
 *
 *
 *
 */
__global__ void mykernel(float *d_data,float *d_result){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    cublasHandle_t handle;
    cublasStatus_t status;
    status=cublasCreate(&handle);
    const float alpha=1.0;
    const float beta=1.0;
    status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[L_power*id],L,&d_data[L_power*id],L,&beta,&d_result[L_power*id],L);
    cublasDestroy(handle);

}

/*
__global__  void nocublas_kernel(float *d_data,float *d_result){
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i,j,k,m;
	for (i=0;i<L;i++){
		for (j=0;j<L;j++){
			d_result[id*L_power+i*L+j]=0.0;
			for (k=0;k<L;k++){
				for (m=0;m<L;m++){
					d_result[id*L_power+i*L+j]+=d_data[L_power*id+i*L+k]*d_data[L_power*id+m*L+j];
					}
				}
			}
		}

}
*/

int main(int argc,char *argv[]){
    int oc;
    FILE *fdata;
    char *datafilename;
    while((oc = getopt(argc, argv,"f:")) !=-1){
        switch(oc)
        {
        case 'f':
            datafilename=optarg;
            break;
        }
    }
//    datafilename="~/..";
    fdata=fopen(datafilename,"rb");
    float *matrix;
    int N=10;
    matrix = new float [L_power*N];
    fseek(fdata,0,SEEK_SET);
    fread(matrix,sizeof(float),N*L_power,fdata);

    float *result;
    result = new float [L_power*N];

    float *d_data;
    float *d_result;

    cudaMalloc((void **) &d_data,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_result,sizeof(float)*N*L_power);

    cudaMemcpy(d_data,matrix,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
    mykernel<<<1,10>>>(d_data,d_result);
//    nocublas_kernel<<<1,10>>>(d_data,d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(result,d_result,sizeof(float)*N*L_power,cudaMemcpyDeviceToHost);

    float *Host_result;
    Host_result = new float [N*L_power];
    for (int i=0;i<N;i++){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,L,L,L,1,&matrix[i*L_power],L,&matrix[i*L_power],L,0,&Host_result[i*L_power],L);
    }

    for (int i=0;i<N;i++){
        float diff=0.0f;
        for (int j=0;j<L_power;j++){
            diff+=(Host_result[i*L_power+j]-result[i*L_power+j])*(Host_result[i*L_power+j]-result[i*L_power+j]);
        }
        printf("diff %d\t%f\n",i,sqrt(diff/(L_power*L_power)));
    }

    cudaFree(d_data);
    cudaFree(d_result);
    delete[] matrix;
    delete[] result;
    delete[] Host_result;
    fclose(fdata);
    return 1;

}
