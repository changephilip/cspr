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


/*
__global__ void mykernel2(float *d_data,float *max_value,cublasHandle_t handle){
	cublasStatus_t status;
	const float alpha=1.0;
	const float beta=0.0;

}
*/

__global__ void simplekernel(float *d_process,float *d_max_value){
    float maxvalue;
    int i;
    maxvalue=d_process[0];
    for (i=0;i<L_power;i++){
        if (maxvalue<d_process[i]){
            maxvalue=d_process[i];
        }
    }
    *d_max_value=maxvalue;
}

__global__ void testlambda(float *d_process){
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	d_process[id]=d_process[id]*1.0;	
}




__global__ void mykernel(float *d_data,float *d_result,cublasHandle_t handle){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    cublasHandle_t handles;
    cublasStatus_t status;
    status=cublasCreate(&handles);
    const float alpha=1.0;
    const float beta=0.0;
    const int si=140;
//    float *C=(float*)malloc(sizeof(float)*si*si);
//    float *C;
//    cudaMalloc((void **)&C,sizeof(float)*si*si); 
    status=cublasSgemm(handles,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,d_data,L,d_data,L,&beta,d_result,L);
//    for (int i=0;i<L_power;i++){
//		C[i]=d_result[i];
//	}
    cublasDestroy(handles);
    //free(C);
//    cudaFree(C);
}
/*
__global__ void mykernel_3(float *d_data,float *d_max_value){
	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha=1.0;
	const float beta=0.0;
	const int si=140;
	int i=0;
	float maxvalue;
	float *C=(float*)malloc(sizeof(float)*si*si);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,d_data,L,d_data,L,&beta,C,L);
	for (i=0;i<L_power;i++){
		


}

*/
//lambdakernel<<<140,140,0,stream[i]>>>(&d_result[L_power*i],&
/*
__global__ void lambdakernel(float *d_process,){
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	int i=blockIdx.x;
	int j=threadIdx.x;
	d_process[id]=d_process[id]+1;
}
*/

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
    const float alpha=1.0;
    const float beta=0.0;
    int N=50;
    matrix = new float [L_power*N];
    fseek(fdata,0,SEEK_SET);
    fread(matrix,sizeof(float),N*L_power,fdata);

    float *result;
    result = new float [L_power*N];

    float *d_data;
    float *d_result;
//    float *d_process[N];
    float max_value[N];
    float *d_max_value;
    cudaMalloc((void **) &d_data,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_result,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_max_value,sizeof(float)*N);

    cudaMemcpy(d_data,matrix,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
//    mykernel<<<1,10>>>(d_data,d_result);
    //    nocublas_kernel<<<1,10>>>(d_data,d_result);
//    cudaDeviceSynchronize();
    //cudaMemcpy(result,d_result,sizeof(float)*N*L_power,cudaMemcpyDeviceToHost);

    cudaStream_t stream[N];
    cublasHandle_t handle[N];
    for(int i=0;i<N;i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&handle[i]);
    }
    for(int i=0;i<N;i++){
        cublasSetStream(handle[i],stream[i]);
    }
    for(int i=0;i<N;i++){
        //cublasSgemm(handle[i],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[L_power*i],L,&d_data[L_power*i],L,&beta,&d_result[L_power*i],L);
	mykernel<<<1,1,0,stream[i]>>>(&d_data[L_power*i],&d_result[L_power*i],handle[i]);
	testlambda<<<L,L,0,stream[i]>>>(&d_result[L_power*i]);
        simplekernel<<<1,1,0,stream[i]>>>(&d_result[L_power*i],&d_max_value[i]);
    //mykernel<<<1,1,0,stream[i]>>>(&d_data[L_power*i],&d_result[L_power*i]);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result,d_result,sizeof(float)*N*L_power,cudaMemcpyDeviceToHost);
    cudaMemcpy(max_value,d_max_value,sizeof(float)*N,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i=0;i<N;i++){
	cublasDestroy(handle[i]);
	cudaStreamDestroy(stream[i]);
	}
    float *Host_result;
    float Host_max[N];
    Host_result = new float [N*L_power];
    for (int i=0;i<N;i++){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,L,L,L,1,&matrix[i*L_power],L,&matrix[i*L_power],L,0,&Host_result[i*L_power],L);
    }

    for (int i=0;i<N;i++){
        Host_max[i]=Host_result[i*L_power];
        for (int j=0;j<L_power;j++){
                if (Host_max[i]<Host_result[i*L_power+j]){
                    Host_max[i]=Host_result[i*L_power+j];
                }
        }
    }
    
    for (int i=0;i<N;i++){
        float diff=0.0f;
        for (int j=0;j<L_power;j++){
            diff+=(Host_result[i*L_power+j]-result[i*L_power+j])*(Host_result[i*L_power+j]-result[i*L_power+j]);
        }
        printf("diff %d\t%f\n",i,sqrt(diff/(L_power*L_power)));
    }
    printf("DIFF %f\t%f\n",Host_result[55],result[55]);    
    for (int i=0;i<N;i++){
        printf("%d\t%f\t%f\n",i,max_value[i],Host_max[i]);
    }

    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_max_value);
    delete[] matrix;
    delete[] result;
    delete[] Host_result;
    fclose(fdata);
    return 1;

}
